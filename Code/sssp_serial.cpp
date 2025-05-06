//g++ -std=c++17 sssp_serial.cpp -o sssp_serial
//./sssp_serial amazon400.txt

#include <iostream>
#include <vector>
#include <queue>
#include <fstream>
#include <sstream>
#include <limits>
#include <chrono>
#include <tuple>
#include <unordered_set>
#include <algorithm>
#include <cstdlib>

using namespace std;
using namespace chrono;

typedef pair<int, int> pii;
const int INF = numeric_limits<int>::max();

// ------------------- Read Edge List --------------------
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

// ------------------- Initial SSSP --------------------
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

// ------------------- Simulate Random Updates --------------------
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

// ------------------- Batch Deletion + Propagation --------------------
void process_deletions(vector<vector<pii>>& adj, vector<int>& dist, vector<int>& parent,
                       const vector<tuple<int, int, int>>& deletions,
                       vector<bool>& affected, vector<bool>& affected_del) {
    for (auto [u, v, w] : deletions) {
        adj[u].erase(remove_if(adj[u].begin(), adj[u].end(),
                               [v](pii edge) { return edge.first == v; }),
                               adj[u].end());
        if (parent[v] == u) {
            affected_del[v] = true;
        }
    }

    queue<int> q;
    for (int i = 0; i < affected_del.size(); ++i) {
        if (affected_del[i]) {
            dist[i] = INF;
            parent[i] = -1;
            q.push(i);
        }
    }

    while (!q.empty()) {
        int curr = q.front(); q.pop();
        for (auto [nbr, wt] : adj[curr]) {
            if (parent[nbr] == curr && !affected_del[nbr]) {
                affected_del[nbr] = true;
                dist[nbr] = INF;
                parent[nbr] = -1;
                q.push(nbr);
            }
        }
    }

    for (int i = 0; i < affected_del.size(); ++i) {
        if (affected_del[i]) {
            affected[i] = true;
        }
    }
}

// ------------------- Batch Insertion + Affected Marking --------------------
void process_insertions(vector<vector<pii>>& adj,
                        const vector<tuple<int, int, int>>& insertions,
                        vector<bool>& affected) {
    for (auto [u, v, w] : insertions) {
        adj[u].push_back({v, w});
        affected[v] = true;
    }
}

// ------------------- Asynchronous Update Loop --------------------
void asynchronous_serial_update(vector<vector<pii>>& adj,
                                vector<int>& dist,
                                vector<int>& parent,
                                vector<bool>& affected) {
    int n = dist.size();
    bool changed = true;
    int iterations = 0;

    while (changed) {
        changed = false;
        iterations++;

        for (int u = 0; u < n; ++u) {
            if (!affected[u] || dist[u] == INF) continue;

            for (auto [v, w] : adj[u]) {
                if (dist[v] > dist[u] + w) {
                    dist[v] = dist[u] + w;
                    parent[v] = u;
                    affected[v] = true;
                    changed = true;
                }
            }

            affected[u] = false;
        }
    }

    cout << "Asynchronous update converged in " << iterations << " iterations.\n";
}

// ------------------- Main --------------------
int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: ./sssp_serial <weighted_edge_list_file>\n";
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

    int num_insertions, num_deletions;
    cout << "Enter number of edge insertions: ";
    cin >> num_insertions;
    cout << "Enter number of edge deletions: ";
    cin >> num_deletions;
    
    vector<tuple<int, int, int>> deletions, insertions;
    simulate_updates(edges, deletions, insertions, num_deletions, num_insertions, n);
    cout << "Simulated " << num_deletions << " deletions and " << num_insertions << " insertions.\n";
    

    // ---------- Full recompute after updates ----------
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
    cout << "Recomputed SSSP after updates in "
         << duration_cast<milliseconds>(end - start).count() << " ms\n";

    // ---------- Reset and apply dynamic update ----------
    vector<vector<pii>> adj_dynamic(n);
    for (auto [u, v, w] : edges) {
        adj_dynamic[u].push_back({v, w});
    }

    recompute_sssp(adj_dynamic, dist, parent);
    vector<bool> affected(n, false), affected_del(n, false);

    start = high_resolution_clock::now();
    process_deletions(adj_dynamic, dist, parent, deletions, affected, affected_del);
    process_insertions(adj_dynamic, insertions, affected);
    asynchronous_serial_update(adj_dynamic, dist, parent, affected);
    end = high_resolution_clock::now();
    cout << "Dynamic update (with flags + async) completed in "
         << duration_cast<milliseconds>(end - start).count() << " ms\n";

    return 0;
}
