// g++ -std=c++17 -fopenmp sssp_openmp.cpp -o sssp_openmp
//./sssp_openmp verybig.txt

#include <iostream>
#include <vector>
#include <queue>
#include <fstream>
#include <sstream>
#include <limits>
#include <chrono>
#include <tuple>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <cstdlib>
#include <omp.h>

using namespace std;
using namespace chrono;

typedef pair<int, int> pii;
const int INF = numeric_limits<int>::max();

vector<tuple<int, int, int>> read_weighted_edge_list(const string& filename, int& max_node) {
    ifstream infile(filename);
    if (!infile.is_open()) {
        cerr << "Failed to open file: " << filename << endl;
        exit(1);
    }

    int u, v, w;
    string line;
    vector<tuple<int, int, int>> edges;
    max_node = 0;

    while (getline(infile, line)) {
        istringstream iss(line);
        if (!(iss >> u >> v >> w)) continue;
        edges.push_back({u, v, w});
        max_node = max(max_node, max(u, v));
    }

    infile.close();
    return edges;
}

void recompute_sssp(vector<vector<pii>>& adj, vector<int>& dist, vector<int>& parent) {
    int n = dist.size();
    dist.assign(n, INF);
    parent.assign(n, -1);
    dist[0] = 0;

    priority_queue<pii, vector<pii>, greater<pii>> pq;
    pq.push({0, 0});

    while (!pq.empty()) {
        auto [d, node] = pq.top(); pq.pop();
        if (d > dist[node]) continue;
        for (auto [nbr, wt] : adj[node]) {
            if (dist[nbr] > dist[node] + wt) {
                dist[nbr] = dist[node] + wt;
                parent[nbr] = node;
                pq.push({dist[nbr], nbr});
            }
        }
    }
}

void initial_sssp(vector<vector<pii>>& adj, vector<int>& dist, vector<int>& parent) {
    auto start = high_resolution_clock::now();
    recompute_sssp(adj, dist, parent);
    auto end = high_resolution_clock::now();
    cout << "Initial SSSP completed in " << duration_cast<milliseconds>(end - start).count() << " ms\n";
}

void simulate_updates(const vector<tuple<int, int, int>>& edges,
                      vector<tuple<int, int, int>>& deletions,
                      vector<tuple<int, int, int>>& insertions,
                      int num_deletions, int num_insertions, int max_node) {
    srand(42);
    unordered_set<int> used_indices;
    while (deletions.size() < num_deletions) {
        int i = rand() % edges.size();
        if (used_indices.count(i)) continue;
        used_indices.insert(i);
        deletions.push_back(edges[i]);
    }

    unordered_set<string> existing_edges;
    for (auto [u, v, w] : edges) {
        existing_edges.insert(to_string(u) + "," + to_string(v));
    }

    while (insertions.size() < num_insertions) {
        int u = rand() % max_node;
        int v = rand() % max_node;
        if (u == v) continue;
        string key = to_string(u) + "," + to_string(v);
        if (existing_edges.count(key)) continue;

        int w = rand() % 10 + 1;
        insertions.push_back({u, v, w});
        existing_edges.insert(key);
    }
}

void mark_subtree(int node, vector<vector<pii>>& adj, vector<int>& parent,
                  vector<int>& dist, vector<bool>& affected, vector<bool>& affected_del) {
    queue<int> q;
    q.push(node);
    affected[node] = true;
    affected_del[node] = true;
    dist[node] = INF;
    parent[node] = -1;

    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (auto [v, w] : adj[u]) {
            if (parent[v] == u && !affected_del[v]) {
                affected_del[v] = true;
                affected[v] = true;
                dist[v] = INF;
                parent[v] = -1;
                q.push(v);
            }
        }
    }
}

void process_deletions(vector<vector<pii>>& adj, vector<int>& dist, vector<int>& parent,
                       const vector<tuple<int, int, int>>& deletions,
                       vector<bool>& affected, vector<bool>& affected_del) {
    for (const auto& [u, v, w] : deletions) {
        adj[u].erase(remove_if(adj[u].begin(), adj[u].end(),
                               [v](pii edge) { return edge.first == v; }),
                               adj[u].end());
        if (parent[v] == u) {
            mark_subtree(v, adj, parent, dist, affected, affected_del);
        }
    }
}

void process_insertions(vector<vector<pii>>& adj, const vector<tuple<int, int, int>>& insertions,
                        vector<int>& dist, vector<int>& parent, vector<bool>& affected,
                        const vector<bool>& affected_del) {
    for (const auto& [u, v, w] : insertions) {
        if (affected_del[v]) continue;
        if (dist[u] != INF && dist[u] + w < dist[v]) {
            adj[u].push_back({v, w});
            dist[v] = dist[u] + w;
            parent[v] = u;
            affected[v] = true;
        }
    }
}

void asynchronous_parallel_update(vector<vector<pii>>& adj,
                                  vector<int>& dist,
                                  vector<int>& parent,
                                  vector<bool>& affected,
                                  int async_depth) {
    int n = dist.size();
    bool changed = true;
    int iterations = 0;

    while (changed) {
        changed = false;
        iterations++;

        #pragma omp parallel for schedule(dynamic)
        for (int u = 0; u < n; ++u) {
            if (!affected[u] || dist[u] == INF) continue;

            queue<pair<int, int>> q;
            q.push({u, 0});
            affected[u] = false;

            while (!q.empty()) {
                auto [curr, depth] = q.front(); q.pop();
                for (auto [v, w] : adj[curr]) {
                    if (dist[v] > dist[curr] + w) {
                        dist[v] = dist[curr] + w;
                        parent[v] = curr;
                        affected[v] = true;
                        changed = true;
                        if (depth + 1 <= async_depth) {
                            q.push({v, depth + 1});
                        }
                    }
                }
            }
        }
    }

    cout << "Asynchronous update converged in " << iterations << " iterations.\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: ./sssp_parallel <weighted_edge_list_file>\n";
        return 1;
    }

    int max_node;
    vector<tuple<int, int, int>> edges = read_weighted_edge_list(argv[1], max_node);
    int n = max_node + 1;

    vector<vector<pii>> adj(n);
    for (auto [u, v, w] : edges) {
        adj[u].push_back({v, w});
    }

    vector<int> dist(n, INF), parent(n, -1);
    initial_sssp(adj, dist, parent);

    int num_insertions, num_deletions, async_depth;
    cout << "Enter number of edge insertions: ";
    cin >> num_insertions;
    cout << "Enter number of edge deletions: ";
    cin >> num_deletions;
    cout << "Enter asynchrony depth: ";
    cin >> async_depth;

    vector<tuple<int, int, int>> deletions, insertions;
    simulate_updates(edges, deletions, insertions, num_deletions, num_insertions, n);
    cout << "Simulated " << num_deletions << " deletions and " << num_insertions << " insertions.\n";

    vector<vector<pii>> adj_recompute = adj;
    for (auto [u, v, w] : deletions) {
        adj_recompute[u].erase(remove_if(adj_recompute[u].begin(), adj_recompute[u].end(),
                                         [v](pii edge) { return edge.first == v; }),
                               adj_recompute[u].end());
    }
    for (auto [u, v, w] : insertions) {
        adj_recompute[u].push_back({v, w});
    }

    auto start = high_resolution_clock::now();
    recompute_sssp(adj_recompute, dist, parent);
    auto end = high_resolution_clock::now();
    cout << "Recomputed SSSP after updates in " << duration_cast<milliseconds>(end - start).count() << " ms\n";

    vector<vector<pii>> adj_dynamic(n);
    for (auto [u, v, w] : edges) {
        adj_dynamic[u].push_back({v, w});
    }

    vector<bool> affected(n, false), affected_del(n, false);

    start = high_resolution_clock::now();
    const int batch_size = 100;
    for (size_t i = 0; i < deletions.size(); i += batch_size) {
        vector<tuple<int, int, int>> batch(deletions.begin() + i, min(deletions.begin() + i + batch_size, deletions.end()));
        process_deletions(adj_dynamic, dist, parent, batch, affected, affected_del);
    }

    for (size_t i = 0; i < insertions.size(); i += batch_size) {
        vector<tuple<int, int, int>> batch(insertions.begin() + i, min(insertions.begin() + i + batch_size, insertions.end()));
        process_insertions(adj_dynamic, batch, dist, parent, affected, affected_del);
    }

    asynchronous_parallel_update(adj_dynamic, dist, parent, affected, async_depth);
    end = high_resolution_clock::now();

    cout << "Dynamic update (with flags + async + OpenMP) completed in "
         << duration_cast<milliseconds>(end - start).count() << " ms\n";

    return 0;
}
