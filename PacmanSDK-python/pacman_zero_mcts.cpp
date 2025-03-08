#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <memory>
#include <vector>

namespace py = pybind11;

py::module copy = py::module::import("copy");
py::module np = py::module::import("numpy");
py::module torch = py::module::import("torch");

py::object get_valid_moves_pacman = py::globals()["get_valid_moves_pacman"];
py::object get_valid_moves_ghost = py::globals()["get_valid_moves_ghost"];
py::object ghostact_int2list = py::globals()["ghostact_int2list"];

class MCTSNode {
public:
    py::object env;
    py::object state;
    py::object state_dict;
    bool done;

    MCTSNode* parent;
    std::vector<std::vector<std::shared_ptr<MCTSNode>>> children_matrix;

    int N;
    py::array_t<float> P_pacman; // size = 5
    py::array_t<float> P_ghost; // size = 125
    double W_pacman;
    double W_ghost;
    double Q_pacman;
    double Q_ghost;
    bool expanded;

    MCTSNode(py::object env_, bool done=false, MCTSNode* parent_=nullptr)
      :done(done), parent(parent_), N(0), W_pacman(0.0), W_ghost(0.0), Q_pacman(0.0), Q_ghost(0.0), expanded(false) {
        
        //py::module copy = py::module::import("copy");
        env = copy.attr("deepcopy")(env_);
        state = env.attr("game_state")();
        state_dict = state.attr("gamestate_to_statedict")();
        
        children_matrix.resize(5, std::vector<std::shared_ptr<MCTSNode>>(125, nullptr));
        
        P_pacman = np.attr("zeros")(5, py::arg("dtype") = np.attr("float32")).cast<py::array_t<float>>();
        P_ghost = np.attr("zeros")(125, py::arg("dtype") = np.attr("float32")).cast<py::array_t<float>>();
    }

    bool is_terminal() const {
        return done;
    }

    bool is_expanded() const {
        return expanded;
    }

    // 扩展当前节点，调用 pacman.predict() 和 ghost.predict() 得到先验概率及价值，并利用外部函数扩展子节点
    std::pair<double, double> expand(py::object pacman, py::object ghost) {
        expanded = true;
        
        py::tuple pacman_out = pacman.attr("predict")(state).cast<py::tuple>();
        py::tuple ghost_out  = ghost.attr("predict")(state).cast<py::tuple>();

        P_pacman = pacman_out[0].attr("cpu")().attr("numpy")().cast<py::array_t<float>>();
        P_ghost  = ghost_out[0].attr("cpu")().attr("numpy")().cast<py::array_t<float>>();

        py::object state_dict_local = state.attr("gamestate_to_statedict")();
        
        py::object pos_pacman = state_dict_local["pacman_coord"];
        py::object pos_ghost  = state_dict_local["ghosts_coord"];
        py::object legal_moves_pacman = get_valid_moves_pacman(pos_pacman, state);
        py::object legal_moves_ghost  = get_valid_moves_ghost(pos_ghost, state);

        for (auto action_pacman_obj:legal_moves_pacman) {
            int a_pacman = action_pacman_obj.cast<int>();
            for (auto action_ghost_obj:legal_moves_ghost) {
                int a_ghost = action_ghost_obj.cast<int>();
                env.attr("restore")(state);
                py::tuple step_out = env.attr("step")(a_pacman, ghostact_int2list(action_ghost_obj), state).cast<py::tuple>();
                bool child_done = step_out[3].cast<bool>();
                children_matrix[a_pacman][a_ghost] = std::make_shared<MCTSNode>(env, child_done, this);
            }
        }
        
        double val_pacman = pacman_out[1].attr("item")().cast<double>();
        double val_ghost  = ghost_out[1].attr("item")().cast<double>();
        
        return {val_pacman, val_ghost};
    }

    std::tuple<int, int, std::shared_ptr<MCTSNode>> select(double c_puct) {
        std::vector<std::pair<int, int>> indices;
        std::vector<double> q_pacman_list, q_ghost_list;
        std::vector<int> n_list;
        std::vector<double> p_pacman_list, p_ghost_list;

        py::object state_dict_local = state.attr("gamestate_to_statedict")();
        py::object pos_pacman = state_dict_local["pacman_coord"];
        py::object pos_ghost  = state_dict_local["ghosts_coord"];
        py::object legal_moves_pacman = get_valid_moves_pacman(pos_pacman, state);
        py::object legal_moves_ghost  = get_valid_moves_ghost(pos_ghost, state);

        auto P_pacman_unchecked = P_pacman.unchecked<1>();
        auto P_ghost_unchecked  = P_ghost.unchecked<1>();

        for (auto action_pacman_obj : legal_moves_pacman) {
            int a_pacman = action_pacman_obj.cast<int>();
            for (auto action_ghost_obj : legal_moves_ghost) {
                int a_ghost = action_ghost_obj.cast<int>();
                std::shared_ptr<MCTSNode> child = children_matrix[a_pacman][a_ghost];
                if (child) {
                    indices.push_back({a_pacman, a_ghost});
                    q_pacman_list.push_back(child->Q_pacman);
                    q_ghost_list.push_back(child->Q_ghost);
                    n_list.push_back(child->N);
                    p_pacman_list.push_back(P_pacman_unchecked(a_pacman));
                    p_ghost_list.push_back(P_ghost_unchecked(a_ghost));
                }
            }
        }
        
        int total_visits = (N>0 ? N : 1);
        double best_score = -1e9;
        int best_index = 0;
        
        for (size_t i = 0; i < n_list.size(); ++i) {
            double bonus = c_puct * std::sqrt(total_visits) / (1 + n_list[i]);
            double score = q_pacman_list[i] + bonus * p_pacman_list[i] +
                           q_ghost_list[i] + bonus * p_ghost_list[i];
            if (score > best_score) {
                best_score = score;
                best_index = i;
            }
        }
        
        int best_action_pacman = indices[best_index].first;
        int best_action_ghost  = indices[best_index].second;
        std::shared_ptr<MCTSNode> best_child = children_matrix[best_action_pacman][best_action_ghost];
        return std::make_tuple(best_action_pacman, best_action_ghost, best_child);
    }

    void update(const std::pair<double, double>& value) {
        double value_pacman = value.first;
        double value_ghost  = value.second;
        N += 1;
        W_pacman += value_pacman;
        W_ghost  += value_ghost;
        Q_pacman = W_pacman/N;
        Q_ghost  = W_ghost/N;
    }
};

class MCTS {
public:
    py::object env;
    py::object state;
    py::object pacman;
    py::object ghost;
    
    double c_puct;
    double temp_inverse;
    int num_simulations;
    
    std::shared_ptr<MCTSNode> root;

    MCTS(py::object env_, py::object pacman_, py::object ghost_,
         double c_puct_, double temperature=1, int num_simulations_=120)
      :env(env_), pacman(pacman_), ghost(ghost_), c_puct(c_puct_), num_simulations(num_simulations_) {
        state = env.attr("game_state")();
        temp_inverse=1.0/temperature;
    }

    std::pair<double, double> search(std::shared_ptr<MCTSNode> node) {
        if (node->is_terminal()) {
            py::object score_obj = node->state_dict["score"];
            py::tuple score_tuple = score_obj.cast<py::tuple>();
            double score_pacman = score_tuple[0].cast<double>();
            double score_ghost  = score_tuple[1].cast<double>();
            return {score_pacman, score_ghost};
        }

        if (!node->is_expanded()) {
            auto value = node->expand(pacman, ghost);
            node->update(value);
            return value;
        }

        int a_pacman, a_ghost;
        std::shared_ptr<MCTSNode> child;
        std::tie(a_pacman, a_ghost, child) = node->select(c_puct);
        auto value = search(child);
        node->update(value);
        return value;
    }

    py::tuple decide(){
        py::module torch = py::module::import("torch");
        py::object device = torch.attr("device")("cuda");
        
        py::object visits_pacman=torch.attr("zeros")(5, py::arg("dtype")=torch.attr("float32"), py::arg("device")=device);
        py::object visits_ghost=torch.attr("zeros")(125, py::arg("dtype")=torch.attr("float32"), py::arg("device")=device);
        double sum_visits = 0.0;

        py::object state_dict_local = state.attr("gamestate_to_statedict")();
        py::object pos_pacman = state_dict_local["pacman_coord"];
        py::object pos_ghost  = state_dict_local["ghosts_coord"];
        py::object legal_moves_pacman = get_valid_moves_pacman(pos_pacman, state);
        py::object legal_moves_ghost  = get_valid_moves_ghost(pos_ghost, state);

        for (auto action_pacman_obj : legal_moves_pacman) {
            int a_pacman = action_pacman_obj.cast<int>();
            for (auto action_ghost_obj : legal_moves_ghost) {
                int a_ghost = action_ghost_obj.cast<int>();
                auto child = root->children_matrix[a_pacman][a_ghost];
                if (child) {
                    float n_val = static_cast<float>(child->N);
                    visits_pacman.attr("put_")(py::make_tuple(a_pacman), py::make_tuple(n_val));
                    visits_ghost.attr("put_")(py::make_tuple(a_ghost), py::make_tuple(n_val));
                    sum_visits += std::pow(n_val, temp_inverse);
                }
            }
        }
        if (sum_visits == 0.0) { sum_visits = 1e-8; }
        py::object prob_pacman = torch.attr("pow")(visits_pacman, temp_inverse) / py::float_(sum_visits);
        py::object prob_ghost  = torch.attr("pow")(visits_ghost, temp_inverse) / py::float_(sum_visits);

        py::object selected_action_pacman = torch.attr("multinomial")(prob_pacman, 1);
        py::object selected_action_ghost  = torch.attr("multinomial")(prob_ghost, 1);
        int action_pacman = selected_action_pacman.attr("item")().cast<int>();
        int action_ghost  = selected_action_ghost.attr("item")().cast<int>();

        auto P_pacman_unchecked = root->P_pacman.unchecked<1>();
        auto P_ghost_unchecked  = root->P_ghost.unchecked<1>();
        if (P_pacman_unchecked(action_pacman) == 0) { action_pacman = 0;}
        if (P_ghost_unchecked(action_ghost) == 0) { action_ghost = 0;}

        py::object Q_pacman_tensor = torch.attr("tensor")(root->Q_pacman,
                                            py::arg("dtype") = torch.attr("float32"), py::arg("device") = device);
        py::object Q_ghost_tensor = torch.attr("tensor")(root->Q_ghost,
                                           py::arg("dtype") = torch.attr("float32"), py::arg("device") = device);
        py::tuple decision_pacman = py::make_tuple(action_pacman, prob_pacman, Q_pacman_tensor);
        py::tuple decision_ghost = py::make_tuple(action_ghost, prob_ghost, Q_ghost_tensor);
        
        return py::make_tuple(decision_pacman, decision_ghost);
    }

    py::tuple run(){
        root=std::make_shared<MCTSNode>(env);
        for (int i=0;i<num_simulations;++i) {
            search(root);
        }
        return decide();
    }
};

PYBIND11_MODULE(mcts_module, m) {
    py::class_<MCTSNode, std::shared_ptr<MCTSNode>>(m, "MCTSNode")
        .def(py::init<py::object, bool, MCTSNode*>(),
             py::arg("env"), py::arg("done") = false, py::arg("parent") = nullptr)
        .def("is_terminal", &MCTSNode::is_terminal)
        .def("is_expanded", &MCTSNode::is_expanded)
        .def("expand", &MCTSNode::expand)
        .def("select", &MCTSNode::select)
        .def("update", &MCTSNode::update);

    py::class_<MCTS>(m, "MCTS")
        .def(py::init<py::object, py::object, py::object, double, double, int>(),
             py::arg("env"), py::arg("pacman"), py::arg("ghost"),
             py::arg("c_puct"), py::arg("temperature") = 1, py::arg("num_simulations") = 120)
        .def("search", &MCTS::search)
        .def("run", &MCTS::run)
        .def("decide", &MCTS::decide);
}