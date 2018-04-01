#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace pybind11::literals;
using namespace std;

class Node;
class Edge;

class State {
    // use string(vector<char>) instead of vector<string> to provide a compact view, map "10" to '1'
    vector<string> histories;
    string player_cards;
    string extra_cards;
    int lord_idx;
    int landscore;
    int control_idx;
    string last_cards;
    int last_category_idx;
    int idx;
    
public:
    explicit State(const py::dict &dict) {
        for (auto item : dict) {
            if (std::string(py::str(item.first)) == "idx") {
                this->idx = (item.second).cast<int>();
            } else if (std::string(py::str(item.first)) == "lord_idx") {
                this->lord_idx = (item.second).cast<int>();
            } else if (std::string(py::str(item.first)) == "control_idx") {
                this->control_idx = (item.second).cast<int>();
            } else if (std::string(py::str(item.first)) == "last_category_idx") {
                this->last_category_idx = (item.second).cast<int>();
            } else if (std::string(py::str(item.first)) == "landscore") {
                this->landscore = (item.second).cast<int>();
            } else if (std::string(py::str(item.first)) == "last_cards") {
                auto last_cards = (item.second).cast<py::list>();
                for (auto card : last_cards) {
                    this->last_cards += std::string(py::str(card))[0];
                }
            }
            // } else if (std::string(py::str(item.first)) == "histories") {
            //     auto histories = (item.second).cast<py::list>();
            //     for (auto l : histories) {
            //         py::list vec = l.cast<py::list>();
            //         if (vec.size() > 0)
            //             printf("vec %s\n", std::string(py::str(vec[0])));
            //     }
            //         // py::print("list item {}: {}"_s.format(index++, l));
            //     // printf("history size %d\n", histories.size());
            // }
        }
    }

    void print() {
        cout << this->last_cards << endl;
    }
};

class Edge {
public:
    int action;
    int n;
    float w;
    float q;
    float prior;
    Node *prev;
    Node *next;
    Edge(Node *node, int action, float prior) {
        this->prev = node;
        this->action = action;
        this->prior = prior;
        this->n = 0;
        this->w = 0;
        this->q = 0;
    }
};


class Node {
public:
    Edge *prev;
    vector<int> actions;
    vector<Edge*> next_edges;
    Node(Edge *edge, vector<int> actions, vector<float> priors) {
        this->prev = edge;
        this->actions = actions;
        for (int i = 0; i < actions.size(); i++) {
            next_edges.push_back(new Edge(this, actions[i], priors[i]));
        }
    }
};

void print_state(py::dict dict) {
    for (auto item : dict) {
        if (std::string(py::str(item.first)) == "idx") {
            printf("idx is %d\n", (item.second).cast<int>());
        } else if (std::string(py::str(item.first)) == "histories") {
            auto histories = (item.second).cast<py::list>();
            for (auto l : histories) {
                py::list vec = l.cast<py::list>();
                if (vec.size() > 0)
                    std::cout << std::string(py::str(vec[0])) << std::endl;
            }
                // py::print("list item {}: {}"_s.format(index++, l));
            // printf("history size %d\n", histories.size());
        }
    }
}


class MCTree {
public:
    explicit MCTree(py::dict dict) {
        State *st = new State(dict);
        st->print();
    }
};