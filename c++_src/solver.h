/***
 * @Author       : luoweiWHUT 1615108374@qq.com
 * @Date         : 2023-11-21 19:06:11
 * @LastEditors  : luoweiWHUT 1615108374@qq.com
 * @LastEditTime : 2023-11-21 19:06:16
 * @FilePath     : \EDA_competition\solver.h
 * @Description  :
 */
#include <iostream>
#include <vector>
#include <algorithm>
#include <ctime>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <numeric>
using namespace std;
class EulerGraph
{
public:
    EulerGraph(const vector<vector<int>> &refs)
    {
        buildGraph(refs);
    }

    void buildGraph(const vector<vector<int>> &refs)
    {
        for (const auto &r : refs)
        {
            if (r[0])
            {
                addNodeAndEdge(r[3], r[5]);
            }
        }
    }

    int getOddNum() const
    {
        int oddNodes = 0;
        for (const auto &entry : graph)
        {
            if (getDegree(entry.first) % 2 != 0)
            {
                oddNodes++;
            }
        }
        return oddNodes;
    }

private:
    unordered_map<int, unordered_set<int>> graph;

    void addNodeAndEdge(int node1, int node2)
    {
        graph[node1].insert(node2);
        graph[node2].insert(node1);
    }

    int getDegree(int node) const
    {
        auto it = graph.find(node);
        if (it != graph.end())
        {
            return it->second.size();
        }
        return 0;
    }
};
class Solver_SA
{
private:
    /*初始化需要的相关参数*/
    const int m, n, p, pinsCodeSize, ref_width, *pinsCode, *array;
    /*设置模拟退火相关参数*/
    double initialTemperature, endTemperature, coolingRate;                     // 初始参数
    int initIterations, iterationsPerTemperature;                               // 初始迭代次数、每个温度下的迭代次数
    vector<vector<vector<int>>> init_sol, best_sol, temp_best_sol, current_sol; // 初始解、最优解、临时最优解、当前解
    double init_score, best_score, temp_best_score, current_score, temperature; // 初始分、最优分数、临时最优分数、当前分数、当前温度
public:
    // 析构函数
    Solver_SA(const int *array, const int m, const int n, const int p, const int *pinsCode, const int pinsCodeSize, const int ref_width);
    // 打印解
    void print_sol(const vector<vector<vector<int>>> &sol);
    // 目标函数
    double getScore(const vector<vector<vector<int>>> &mosListEncode1);
    // 邻域解生成方法
    vector<vector<vector<int>>> update_sol(const vector<vector<vector<int>>> &s_old, int action);
    // 爬山法初始化温度
    void init_Temperature();
    // 将当前状态重置回初始状态
    void reset()
    {
        temp_best_sol = init_sol, current_sol = init_sol;
        temp_best_score = init_score, current_score = init_score;
    }
    vector<vector<vector<int>>> run_SA(); // 运行模拟退火算法，并返回最优解
    void annealing();                     // 进行退火优化(一轮)
};
extern "C" __declspec(dllexport) int *run_SA(const int *array, const int m, const int n, const int p, const int *pinsCode, const int pinsCodeSize, const int ref_width);
extern "C" __declspec(dllexport) void destroy_array(int *result);