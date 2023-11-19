/***
 * @Author       : luoweiWHUT 1615108374@qq.com
 * @Date         : 2023-11-18 14:56:20
 * @LastEditors  : luoweiWHUT 1615108374@qq.com
 * @LastEditTime : 2023-11-18 15:08:35
 * @FilePath     : \EDA_competition\solver.cpp
 * @Description  :
 */
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <ctime>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <numeric>
using namespace std;
void print_vector(vector<vector<vector<int>>> &s_vector, int m, int n, int p)
{
    std::cout << "[";
    for (int i = 0; i < m; ++i)
    {
        std::cout << "[";
        for (int j = 0; j < n; ++j)
        {
            std::cout << "[";
            for (int k = 0; k < p; ++k)
            {
                std::cout << s_vector[i][j][k] << ", ";
            }
            std::cout << "],";
        }
        std::cout << "]," << endl;
    }
    std::cout << "]" << endl;
}

class EulerGraph
{
public:
    EulerGraph(const std::vector<std::vector<int>> &refs)
    {
        buildGraph(refs);
    }

    void buildGraph(const std::vector<std::vector<int>> &refs)
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
    std::unordered_map<int, std::unordered_set<int>> graph;

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
double getScore(const std::vector<std::vector<std::vector<int>>> &mosListEncode1, const int *pinsCode, const int &pinsCodeSize, const int &ref_width1)
{
    // Deep copy mosListEncode1
    auto mosListEncode = mosListEncode1;

    // 计算x坐标并设置symmetric
    double symmetric = 10.0;
    int unit_x = 0;

    for (size_t i = 0; i < mosListEncode[0].size(); ++i)
    {
        if (i == 0)
        {
            unit_x = 0;
        }
        else
        {
            if ((mosListEncode[0][i - 1][0] * mosListEncode[0][i][0] == 0 || mosListEncode[0][i - 1][5] == mosListEncode[0][i][2]) &&
                (mosListEncode[1][i - 1][0] * mosListEncode[1][i][0] == 0 || mosListEncode[1][i - 1][5] == mosListEncode[1][i][2]))
            {
                unit_x += 1;
            }
            else
            {
                unit_x += 2;
            }
        }
        mosListEncode[0][i].insert(mosListEncode[0][i].begin() + 1, unit_x);
        mosListEncode[1][i].insert(mosListEncode[1][i].begin() + 1, unit_x);

        if (mosListEncode[0][i][0] * mosListEncode[1][i][0] == 0 && mosListEncode[0][i][0] + mosListEncode[1][i][0] > 0)
        {
            symmetric -= 1.0;
        }
    }
    // 设置drc
    double drc = 10.0;
    for (size_t i = 0; i < mosListEncode[0].size(); ++i)
    {
        if (0 < i && i < mosListEncode[0].size() - 1)
        {
            if (mosListEncode[0][i + 1][1] - mosListEncode[0][i - 1][1] == 2 &&
                mosListEncode[0][i - 1][0] && mosListEncode[0][i][0] && mosListEncode[0][i + 1][0] &&
                mosListEncode[0][i][mosListEncode[0][i].size() - 1] < mosListEncode[0][i - 1][mosListEncode[0][i - 1].size() - 1] &&
                mosListEncode[0][i][mosListEncode[0][i].size() - 1] < mosListEncode[0][i + 1][mosListEncode[0][i + 1].size() - 1])
            {
                drc -= 10.0;
            }
            if (mosListEncode[1][i + 1][1] - mosListEncode[1][i - 1][1] == 2 &&
                mosListEncode[1][i - 1][0] && mosListEncode[1][i][0] && mosListEncode[1][i + 1][0] &&
                mosListEncode[1][i][mosListEncode[1][i].size() - 1] < mosListEncode[1][i - 1][mosListEncode[1][i - 1].size() - 1] &&
                mosListEncode[1][i][mosListEncode[1][i].size() - 1] < mosListEncode[1][i + 1][mosListEncode[1][i + 1].size() - 1])
            {
                drc -= 10.0;
            }
        }
    }
    // 设置width
    int width = unit_x + 1;

    // 计算线网位置
    std::vector<std::vector<int>> mosList = mosListEncode[0];
    mosList.insert(mosList.end(), mosListEncode[1].begin(), mosListEncode[1].end());
    std::unordered_map<int, std::vector<double>> net_positions;

    for (const auto &mos : mosList)
    {
        auto nets = std::vector<int>(mos.begin() + 3, mos.begin() + 6);
        auto positions = std::vector<double>{static_cast<double>(mos[1]) - 0.5, static_cast<double>(mos[1]), static_cast<double>(mos[1]) + 0.5};
        for (size_t i = 0; i < nets.size(); ++i)
        {
            net_positions[nets[i]].push_back(positions[i]);
        }
    }
    // 设置pin_access
    std::vector<double> pin_coords;
    for (auto &entry : net_positions)
    {
        sort(entry.second.begin(), entry.second.end());
        if (std::find(pinsCode, pinsCode + pinsCodeSize, entry.first) != pinsCode + pinsCodeSize)
        {
            pin_coords.push_back(entry.second[0]);
            double max_distance = 0.0;
            for (double pos : entry.second)
            {
                double distance = 0.0;
                std::vector<double> another_pos;
                for (const auto &another_entry : net_positions)
                {
                    // 使用 std::find 查找 pinsCode 数组中是否包含 another_entry.first
                    auto pinsCodeIterator = std::find(pinsCode, pinsCode + pinsCodeSize, another_entry.first);

                    // 检查是否找到，并且 another_entry.first 不等于 entry.first
                    if (pinsCodeIterator != pinsCode + pinsCodeSize && *pinsCodeIterator != entry.first)
                    {
                        // 使用 another_entry.second.begin() 和 another_entry.second.end() 插入到 another_pos
                        another_pos.insert(another_pos.end(), another_entry.second.begin(), another_entry.second.end());
                    }
                }

                if (!another_pos.empty())
                {
                    std::sort(another_pos.begin(), another_pos.end());

                    if (another_pos.front() > pos)
                    {
                        distance = std::abs(another_pos.front() - pos);
                    }
                    else if (another_pos.back() < pos)
                    {
                        distance = std::abs(another_pos.back() - pos);
                    }
                    else
                    {
                        for (size_t i = 0; i < another_pos.size() - 1; ++i)
                        {
                            if (another_pos[i] < pos && pos < another_pos[i + 1])
                            {
                                distance = std::min(std::abs(another_pos[i] - pos), std::abs(another_pos[i + 1] - pos));
                                break;
                            }
                        }
                    }
                }

                if (distance > max_distance)
                {
                    max_distance = distance;
                    pin_coords.back() = pos;
                }
            }
        }
    }

    std::sort(pin_coords.begin(), pin_coords.end());

    double pin_access = 1.0;

    if (pin_coords.empty() || pin_coords.size() == 1)
    {
        pin_access = 1.0;
    }
    else
    {
        std::vector<double> pin_spacing;
        double left_spacing = pin_coords.front() + 0.5;
        double right_spacing = width - 0.5 - pin_coords.back();

        if (left_spacing > 1)
        {
            pin_spacing.push_back(left_spacing / width);
        }

        if (right_spacing > 1)
        {
            pin_spacing.push_back(right_spacing / width);
        }

        for (size_t i = 0; i < pin_coords.size() - 1; ++i)
        {
            pin_spacing.push_back((pin_coords[i + 1] - pin_coords[i]) / width);
        }
        double sum = std::accumulate(pin_spacing.begin(), pin_spacing.end(), 0.0,
                                     [](double acc, double val)
                                     { return acc + val; });

        double meanSpacing = sum / pin_spacing.size();
        double sumSquaredDifferences = std::accumulate(pin_spacing.begin(), pin_spacing.end(), 0.0,
                                                       [meanSpacing](double acc, double val)
                                                       { return acc + std::pow(val - meanSpacing, 2); });

        pin_access = std::sqrt(sumSquaredDifferences / pin_spacing.size());
    }
    // 设置bbox
    double bbox = 0.0;
    net_positions.erase(0); // VSS
    net_positions.erase(1); // VDD
    for (auto &entry : net_positions)
    {
        bbox += entry.second.back() - entry.second.front();
    }
    // 设置ref_width
    EulerGraph upperGraph(mosListEncode[1]);
    EulerGraph lowerGraph(mosListEncode[0]);
    double min_gap = std::max(0.0, (upperGraph.getOddNum() + lowerGraph.getOddNum() - 4) / 2.0);
    double ref_width = (min_gap + ref_width1) / 2.0;
    // cout << "width:" << width << " bbox:" << bbox << "ref_width" << ref_width << "pin_access" << pin_access
    //      << " symmetric" << symmetric << " drc" << drc << endl;
    // 设置score
    double ws = 40.0 * (1.0 - (width - ref_width) / (ref_width + 20.0));
    double bs = std::min(20.0, 20.0 * (1.0 - (bbox - ref_width * (pinsCodeSize - 1)) / 60.0));
    double ps = 10.0 * (1.0 - pin_access);
    double rs = 10 / (1 + std::exp(0 / 3600 - 1));
    double score = symmetric + drc + ws + bs + ps + rs;
    return score;
}
vector<vector<vector<int>>> update_vector(vector<vector<vector<int>>> &s_old, int action)
{
    // Create a new vector to store the updated solution
    vector<vector<vector<int>>> s_new = s_old;
    if (action == 0)
    {
        // Randomly select two indices and move the pair to a new position
        int a = rand() % s_new[0].size();
        int b = rand() % s_new[0].size();
        iter_swap(s_new[0].begin() + a, s_new[0].begin() + b);
        iter_swap(s_new[1].begin() + a, s_new[1].begin() + b);
    }
    else if (action == 1)
    {
        // Randomly select two indices and swap the pairs
        int a = rand() % s_new[0].size();
        int b = rand() % s_new[0].size();
        swap(s_new[0][a], s_new[0][b]);
        swap(s_new[1][a], s_new[1][b]);
    }
    else if (action == 2)
    {
        // Swap elements with the same channel type
        int channel_type = rand() % 2;
        int a = rand() % s_new[0].size();
        vector<int> indices;
        for (int i = 0; i < s_new[channel_type].size(); ++i)
        {
            if (s_new[channel_type][i][3] == s_new[channel_type][a][3])
            {
                indices.push_back(i);
            }
        }
        int b = indices[rand() % indices.size()];
        swap(s_new[channel_type][a], s_new[channel_type][b]);
    }
    else if (action == 3)
    {
        // Randomly rotate a pipe
        int a = rand() % s_new[0].size();
        int channel_type = rand() % 2;
        swap(s_new[channel_type][a][2], s_new[channel_type][a][4]);
    }
    // Add more conditions as needed
    return s_new;
}
void init_SA(vector<vector<vector<int>>> s_vector, double current_score, vector<vector<vector<int>>> &best_sol, double &best_score, double &initialTemperature, double &endTemperature, const int &iterationsPerTemperature, const int *pinsCode, const int &pinsCodeSize, const int &ref_width)
{
    int i = 0;
    double sum_loss = 0;
    while (i < iterationsPerTemperature)
    {
        int action = rand() % 4;
        vector<vector<vector<int>>> new_s_vector = update_vector(s_vector, action);
        double new_score = getScore(new_s_vector, pinsCode, pinsCodeSize, ref_width);
        double scoreDelta = new_score - current_score;
        // 如果新解更差：

        if (scoreDelta < 0)
        {
            sum_loss -= scoreDelta;
        }
        else
        {
            s_vector = new_s_vector;
            current_score = new_score;
            // 更新最优解
            if (new_score > best_score)
            {
                best_score = new_score;
                best_sol = new_s_vector;
            }
        }
        i++;
    }
    initialTemperature = 2 * sum_loss / iterationsPerTemperature;
    endTemperature = 0.01 * initialTemperature;
}
extern "C"
{
    int *run_SA(int *array, const int m, const int n, const int p, const int *pinsCode, const int pinsCodeSize, const int ref_width)
    {
        srand(time(nullptr)); // 初始化随机数种子
        /*读入并设置初始解*/
        vector<vector<vector<int>>> s_vector;
        s_vector.resize(m, std::vector<std::vector<int>>(n, std::vector<int>(p)));
        cout << "数组的形状为:m=" << m << " n=" << n << " p=" << p << endl;
        if (array == nullptr)
        {
            std::cerr << "Error: Null pointer passed to print_array function." << std::endl;
        }
        // Check if the array size is consistent with the specified dimensions
        if (m * n * p != distance(array, array + m * n * p))
        {
            // Handle the error as needed, e.g., throw an exception
            throw invalid_argument("Array size does not match specified dimensions");
        }
        if (m <= 0 || n <= 0 || p <= 0)
        {
            // Handle the error as needed, e.g., throw an exception
            throw invalid_argument("Array size 含有负数");
        }
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                for (int k = 0; k < p; ++k)
                {
                    s_vector[i][j][k] = array[i * n * p + j * p + k];
                }
            }
        }
        double current_score = getScore(s_vector, pinsCode, pinsCodeSize, ref_width);

        /*设置模拟退火相关参数*/
        double initialTemperature = 100;                 // 初始温度
        double endTemperature = 1;                       // 初始温度
        const double coolingRate = 0.95;                 // 退火速率
        const int iterationsPerTemperature = 10 * m * n; // 每个温度的迭代次数
        vector<vector<vector<int>>> best_sol = s_vector;
        double best_score = current_score;
        init_SA(s_vector, current_score, best_sol, best_score, initialTemperature, endTemperature, iterationsPerTemperature, pinsCode, pinsCodeSize, ref_width);
        double temperature = initialTemperature;
        while (temperature > endTemperature)
        {
            for (int i = 0; i < iterationsPerTemperature; ++i)
            {
                int action = rand() % 4;
                vector<vector<vector<int>>> new_s_vector = update_vector(s_vector, action);
                double new_score = getScore(new_s_vector, pinsCode, pinsCodeSize, ref_width);
                double scoreDelta = new_score - current_score;
                // 如果新解更优或以一定概率接受劣解
                if (scoreDelta > 0 || exp(scoreDelta / temperature) > (rand() / double(RAND_MAX)))
                {
                    s_vector = new_s_vector;
                    current_score = new_score;
                    // 更新最优解
                    if (new_score > best_score)
                    {
                        best_score = new_score;
                        best_sol = new_s_vector;
                    }
                }
            }
            // 降低温度
            temperature *= coolingRate;
        }
        /*输出结果*/
        // 使用动态分配内存创建三维数组
        int size = m * n * p;
        int *s = new int[size];
        // Copy values from the array to the 3D vector
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                for (int k = 0; k < p; ++k)
                {
                    s[i * n * p + j * p + k] = best_sol[i][j][k];
                }
            }
        }
        cout << "Best Score: " << best_score << endl;
        cout << "***********************************************************************************" << endl;
        return s;
    }
    // 销毁动态分配的三维数组
    void destroy_vector(int *s)
    {
        delete[] s;
    }
    void print_array(int *array, int size)
    {
        for (int i = 0; i < size; ++i)
        {
            std::cout << array[i] << " ";
        }
        std::cout << std::endl;
    }
}
int main()
{
    // 测试case:SEDFD2
    int array[] = {39, 1, 13, 8, 32, 155, 61, 1, 29, 8, 1, 120, 35, 1, 13, 2, 31, 125, 36, 1, 13, 2, 31, 125, 41, 1, 9, 24, 4, 200, 59, 1, 21, 24, 10, 150, 60, 1, 21, 24, 10, 145, 40, 1, 32, 5, 1, 155, 42, 1, 7, 5, 1, 170, 43, 1, 7, 5, 1, 170, 56, 1, 10, 5, 1, 120, 0, 1, 30, 5, 0, 120, 0, 1, 30, 5, 0, 120, 65, 1, 17, 28, 1, 170, 49, 1, 5, 21, 1, 170, 51, 1, 15, 21, 1, 170, 52, 1, 15, 21, 1, 170, 0, 1, 5, 21, 0, 135, 0, 1, 5, 21, 0, 135, 0, 1, 5, 21, 0, 135, 44, 1, 12, 27, 1, 120, 47, 1, 24, 26, 1, 165, 48, 1, 24, 26, 1, 165, 53, 1, 9, 26, 12, 120, 57, 1, 21, 26, 27, 150, 58, 1, 21, 26, 27, 145, 45, 1, 27, 9, 1, 150, 46, 1, 27, 9, 1, 145, 37, 1, 26, 20, 1, 165, 38, 1, 26, 20, 1, 165, 54, 1, 25, 16, 1, 145, 55, 1, 25, 16, 1, 140, 62, 1, 11, 16, 1, 145, 63, 1, 11, 16, 1, 140, 33, 1, 31, 29, 1, 125, 34, 1, 31, 29, 1, 125, 32, 1, 28, 19, 1, 170, 64, 1, 4, 11, 17, 170, 50, 1, 4, 13, 25, 170, 1, 0, 3, 8, 0, 220, 30, 0, 29, 8, 0, 220, 2, 0, 13, 2, 3, 155, 0, 0, 13, 2, 31, 125, 3, 0, 21, 24, 27, 130, 4, 0, 21, 24, 27, 130, 15, 0, 9, 24, 23, 120, 5, 0, 7, 5, 0, 140, 6, 0, 7, 5, 0, 135, 7, 0, 7, 5, 0, 140, 8, 0, 7, 5, 0, 135, 24, 0, 22, 5, 0, 155, 31, 0, 30, 5, 0, 120, 9, 0, 14, 28, 0, 140, 10, 0, 15, 21, 0, 140, 11, 0, 15, 21, 0, 135, 12, 0, 15, 21, 0, 140, 13, 0, 15, 21, 0, 135, 18, 0, 5, 21, 0, 140, 19, 0, 5, 21, 0, 135, 14, 0, 23, 27, 0, 120, 16, 0, 24, 26, 0, 140, 17, 0, 9, 26, 18, 190, 25, 0, 21, 26, 30, 120, 0, 0, 21, 26, 27, 145, 0, 0, 21, 26, 27, 145, 20, 0, 27, 9, 0, 140, 0, 0, 27, 9, 1, 145, 21, 0, 26, 20, 0, 140, 0, 0, 26, 20, 1, 165, 22, 0, 18, 16, 14, 140, 26, 0, 11, 16, 0, 160, 0, 0, 11, 16, 1, 140, 0, 0, 11, 16, 1, 140, 23, 0, 13, 29, 22, 155, 0, 0, 31, 29, 1, 125, 27, 0, 28, 19, 0, 140, 28, 0, 6, 11, 0, 160, 29, 0, 18, 13, 6, 140};
    int pinsCode[] = {8, 16, 20, 19, 2, 15, 7};
    const int m = 2, n = 39, p = 6, pinsCodeSize = 7, ref_width = 50;
    // const int n = arraySize / 12;
    int *array_result = run_SA(array, m, n, p, pinsCode, pinsCodeSize, ref_width);
    // print_array(array_result, m * n * p);
    destroy_vector(array_result);
    return 0;
}