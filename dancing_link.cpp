#include "dancing_link.h"


Object *createHead(const uint8_t *mat, const py::array_t<uint8_t, py::array::c_style | py::array::forcecast> &mask, int rows, int cols, vector<Object*> &objs) {
    auto h = new ColumnObject();
    objs.push_back(h);
    vector<Object*> rows_obj(rows, nullptr);

    auto iter_c = h;
    auto mask_ptr = mask.unchecked<1>();
    for (int i = 0; i < cols; i++) {
        if (mask_ptr[i] == 0) continue;
        auto c = new ColumnObject();
        objs.push_back(c);
        iter_c->r = c;
        c->l = iter_c;
        c->c = c;

        int one_cnt = 0;
        Object *iter_r = c;
        for (int j = 0; j < rows; j++) {
            if (mat[j * cols + i] > 0) {
                one_cnt++;
                auto r = new DataObject();
                objs.push_back(r);
                iter_r->d = r;
                r->u = iter_r;
                r->c = c;
                r->r_idx = j;

                iter_r = r;

                if (!rows_obj[j]) {
                    rows_obj[j] = r;
                    r->r = r;
                    r->l = r;
                }
                else {
                    rows_obj[j]->r->l = r;
                    r->r = rows_obj[j]->r;
                    rows_obj[j]->r = r;
                    r->l = rows_obj[j];
                    rows_obj[j] = r;
                }
            }
        }
        iter_r->d = c;
        c->u = iter_r;
        c->s = one_cnt;
        char name[128];
        sprintf(name, "Col %d", i);
        c->name = name;

        iter_c = c;
    }
    iter_c->r = h;
    h->l = iter_c;
    return h;
}

void coverCol(Object* c) {
    c->l->r = c->r;
    c->r->l = c->l;

    auto i = c->d;
    while (i != c) {
        auto j = i->r;
        while (j != i) {
            j->u->d = j->d;
            j->d->u = j->u;
            static_cast<ColumnObject*>(j->c)->s -= 1;

            j = j->r;
        }
        i = i->d;
    }
}

void uncoverCol(Object* c) {
    auto i = c->u;
    while (i != c) {
        auto j = i->l;
        while (j != i) {
            static_cast<ColumnObject*>(j->c)->s += 1;
            j->d->u = j;
            j->u->d = j;

            j = j->l;
        }
        i = i->u;
    }

    c->r->l = c;
    c->l->r = c;
}

void search(Object *head, vector<int> &path, py::list &results) {
    if (head->r == head) {
        // for (const auto &p : path) {
        //     cout << p << ", ";
        // }
        // cout << endl;
        results.append(py::array(path.size(), path.data()));
        return;
    }

    int min_cnt = std::numeric_limits<int>::max();
    ColumnObject *cand_c;
    for (auto r = head->r; r != head;  r = r->r) {
        auto c = static_cast<ColumnObject*>(r);
        int cnt = c->s;
        if (cnt < min_cnt) {
            min_cnt = cnt;
            cand_c = c;
        }
    }

    coverCol(cand_c);
    for (auto r = cand_c->d; r != cand_c; r = r->d) {
        path.push_back(static_cast<DataObject*>(r)->r_idx);
        
        for (auto j = r->r; j != r; j = j->r) {
            coverCol(j->c);
        }
        search(head, path, results);
        for (auto j = r->l; j != r; j = j->l) {
            uncoverCol(j->c);
        }
        path.pop_back();
    }
    uncoverCol(cand_c);
}

py::list get_combinations_nosplit(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> arr, py::array_t<uint8_t, py::array::c_style | py::array::forcecast> mask) {
    const uint8_t *mat = arr.data();
    vector<Object*> objects;
    ColumnObject *head = static_cast<ColumnObject *>(createHead(mat, mask, arr.shape(0), arr.shape(1), objects));
    vector<int> path;
    py::list results;
    search(head, path, results);
    for (auto obj : objects) {
        if (obj) delete obj;
    }
    return results;
}

void helper(const uint8_t *mat, vector<uint8_t> &target, int r_idx, int n_rows, vector<int> &path, py::list &results) {
    if (accumulate(target.begin(), target.end(), (int)0) == 0) {
        results.append(py::array(path.size(), path.data()));
        return;
    }
    for (int i = r_idx; i < n_rows; i++) {
        bool valid = true;
        for (int j = 0; j < 15; j++) {
            if (mat[i * 15 + j] > target[j]) {
                valid = false;
                break;
            }
        }
        if (!valid) continue;
        for (int j = 0; j < 15; j++) {
            target[j] -= mat[i * 15 + j];
        }
        path.push_back(i);
        helper(mat, target, i, n_rows, path, results);
        path.pop_back();
        for (int j = 0; j < 15; j++) {
            target[j] += mat[i * 15 + j];
        } 
    }
}

py::list get_combinations_recursive(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> arr,
        py::array_t<uint8_t, py::array::c_style | py::array::forcecast> target) {
    const uint8_t *mat = arr.data();
    vector<uint8_t> cnt(target.data(), target.data() + target.size());
    py::list results;
    vector<int> path;
    helper(mat, cnt, 0, arr.size() / 15, path, results);
    return results;
}