#include <iostream>
#include <vector>
#include <functional>
#include <queue>
using namespace std;



double reLU(double x){
    return max(0.0,x);
}

double no_change(double x){
    return x;
}


int main(){
    vector<function<double(double)>> nodes; 
    vector<vector<int>> edges; 
    nodes.push_back(no_change);
    nodes.push_back(reLU);
    nodes.push_back(no_change);
    edges.push_back({1});
    edges.push_back({2});
    edges.push_back({});
    double value = 0.57;
    queue<int> q;
    q.push(0);
    while(!q.empty()){
        int node = q.front();
        value = nodes[node](value);
        for(int connected_node : edges[node]){
            q.push(connected_node);
        }
        q.pop();
    }
    cout << value;

}