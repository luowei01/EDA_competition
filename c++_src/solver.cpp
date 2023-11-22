/***
 * @Author       : luoweiWHUT 1615108374@qq.com
 * @Date         : 2023-11-18 14:56:20
 * @LastEditors  : luoweiWHUT 1615108374@qq.com
 * @LastEditTime : 2023-11-18 15:08:35
 * @FilePath     : \EDA_competition\solver.cpp
 * @Description  :
 */
#include "solver.h"
Solver_SA::Solver_SA(const int *array, const int m, const int n, const int p, const int *pinsCode, const int pinsCodeSize, const int ref_width)
    : array(array), m(m), n(n), p(p), pinsCode(pinsCode), pinsCodeSize(pinsCodeSize), ref_width(ref_width)
{
    cout << "晶体管经过拆分和虚拟配对后的数量为:" << m << " x " << n << endl;
    if (array == nullptr)
    {
        cerr << "Error: Null pointer passed to print_array function." << endl;
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
    init_sol.resize(m, vector<vector<int>>(n, vector<int>(p)));
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            for (int k = 0; k < p; ++k)
            {
                init_sol[i][j][k] = array[i * n * p + j * p + k];
            }
        }
    }
    init_score = getScore(init_sol);
    best_sol = init_sol, temp_best_sol = init_sol, current_sol = init_sol;
    best_score = init_score, temp_best_score = init_score, current_score = init_score;
    coolingRate = 0.95, initIterations = 10 * m * n, iterationsPerTemperature = 20 * m * n;
}
void Solver_SA::print_sol(const vector<vector<vector<int>>> &sol)
{
    cout << "[";
    for (int i = 0; i < m; ++i)
    {
        cout << "[";
        for (int j = 0; j < n; ++j)
        {
            cout << "[";
            for (int k = 0; k < p; ++k)
            {
                cout << sol[i][j][k] << ", ";
            }
            cout << "],";
        }
        cout << "]," << endl;
    }
    cout << "]" << endl;
    cout << "score:" << getScore((sol)) << endl;
}
double Solver_SA::getScore(const vector<vector<vector<int>>> &mosListEncode1)
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
    vector<vector<int>> mosList = mosListEncode[0];
    mosList.insert(mosList.end(), mosListEncode[1].begin(), mosListEncode[1].end());
    unordered_map<int, vector<double>> net_positions;

    for (const auto &mos : mosList)
    {
        auto nets = vector<int>(mos.begin() + 3, mos.begin() + 6);
        auto positions = vector<double>{static_cast<double>(mos[1]) - 0.5, static_cast<double>(mos[1]), static_cast<double>(mos[1]) + 0.5};
        for (size_t i = 0; i < nets.size(); ++i)
        {
            net_positions[nets[i]].push_back(positions[i]);
        }
    }
    // 设置pin_access
    vector<double> pin_coords;
    for (auto &entry : net_positions)
    {
        sort(entry.second.begin(), entry.second.end());
        if (find(pinsCode, pinsCode + pinsCodeSize, entry.first) != pinsCode + pinsCodeSize)
        {
            pin_coords.push_back(entry.second[0]);
            double max_distance = 0.0;
            for (double pos : entry.second)
            {
                double distance = 0.0;
                vector<double> another_pos;
                for (const auto &another_entry : net_positions)
                {
                    // 使用 find 查找 pinsCode 数组中是否包含 another_entry.first
                    auto pinsCodeIterator = find(pinsCode, pinsCode + pinsCodeSize, another_entry.first);

                    // 检查是否找到，并且 another_entry.first 不等于 entry.first
                    if (pinsCodeIterator != pinsCode + pinsCodeSize && *pinsCodeIterator != entry.first)
                    {
                        // 使用 another_entry.second.begin() 和 another_entry.second.end() 插入到 another_pos
                        another_pos.insert(another_pos.end(), another_entry.second.begin(), another_entry.second.end());
                    }
                }

                if (!another_pos.empty())
                {
                    sort(another_pos.begin(), another_pos.end());

                    if (another_pos.front() > pos)
                    {
                        distance = abs(another_pos.front() - pos);
                    }
                    else if (another_pos.back() < pos)
                    {
                        distance = abs(another_pos.back() - pos);
                    }
                    else
                    {
                        for (size_t i = 0; i < another_pos.size() - 1; ++i)
                        {
                            if (another_pos[i] < pos && pos < another_pos[i + 1])
                            {
                                distance = min(abs(another_pos[i] - pos), abs(another_pos[i + 1] - pos));
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

    sort(pin_coords.begin(), pin_coords.end());

    double pin_access = 1.0;

    if (pin_coords.empty() || pin_coords.size() == 1)
    {
        pin_access = 1.0;
    }
    else
    {
        vector<double> pin_spacing;
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
        double sum = accumulate(pin_spacing.begin(), pin_spacing.end(), 0.0);

        double meanSpacing = sum / pin_spacing.size();
        double sumSquaredDifferences = 0.0;
        std::for_each(pin_spacing.begin(), pin_spacing.end(), [&](const double d)
                      { sumSquaredDifferences += pow(d - meanSpacing, 2); });
        pin_access = sqrt(sumSquaredDifferences / pin_spacing.size());
    }
    // 设置bbox
    double bbox = 0.0;
    net_positions.erase(0); // VSS
    net_positions.erase(1); // VDD
    for (auto &entry : net_positions)
    {
        // if (entry.first == 0 || entry.first == 1)
        //     continue;
        bbox += entry.second.back() - entry.second.front();
    }
    // 设置ref_width
    EulerGraph upperGraph(mosListEncode[1]);
    EulerGraph lowerGraph(mosListEncode[0]);
    double min_gap = max(0.0, (upperGraph.getOddNum() + lowerGraph.getOddNum() - 4) / 2.0);
    double ref_width1 = (min_gap + ref_width) / 2.0;
    // cout << "width:" << width << " bbox:" << bbox << "ref_width" << ref_width << "pin_access" << pin_access
    //  << " symmetric" << symmetric << " drc" << drc << endl;
    // 设置score
    double ws = 40.0 * (1.0 - (width - ref_width1) / (ref_width1 + 20.0));
    double bs = min(20.0, 20.0 * (1.0 - (bbox - ref_width1 * (pinsCodeSize - 1)) / 60.0));
    double ps = 10.0 * (1.0 - pin_access);
    double rs = 10 / (1 + exp(0 / 3600 - 1));
    double score = symmetric + drc + ws + bs + ps + rs;
    return score;
}
vector<vector<vector<int>>> Solver_SA::update_sol(const vector<vector<vector<int>>> &s_old, int action)
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
void Solver_SA::init_Temperature()
{
    int i = 0;
    double sum_loss = 0;
    while (i < initIterations)
    {
        int action = rand() % 4;
        vector<vector<vector<int>>> new_sol = update_sol(current_sol, action);
        double new_score = getScore(new_sol);
        double scoreDelta = new_score - current_score;
        // 如果新解更差：
        if (scoreDelta < 0)
        {
            sum_loss -= scoreDelta;
        }
        else
        {
            current_sol = new_sol;
            current_score = new_score;
            // 更新最优解
            if (new_score > best_score)
            {
                best_score = new_score;
                best_sol = new_sol;
            }
        }
        i++;
    }
    initialTemperature = 2 * sum_loss / initIterations;
    endTemperature = 0.01 * initialTemperature;
}
void Solver_SA::annealing()
{
    temperature = initialTemperature;
    while (temperature > endTemperature)
    {
        if (best_score > 97)
        {
            break;
        }
        for (int i = 0; i < iterationsPerTemperature; ++i)
        {
            int action = rand() % 4;
            vector<vector<vector<int>>> new_sol = update_sol(current_sol, action);
            double new_score = getScore(new_sol);
            double scoreDelta = new_score - current_score;
            // 如果新解更优或以一定概率接受劣解
            if (scoreDelta > 0 || exp(scoreDelta / temperature) > (rand() / double(RAND_MAX)))
            {
                current_sol = new_sol;
                current_score = new_score;
                // 更新最优解
                if (new_score > temp_best_score)
                {
                    temp_best_score = new_score;
                    temp_best_sol = new_sol;
                }
            }
        }
        // 降低温度
        temperature *= coolingRate;
    }
    if (best_score < temp_best_score)
    {
        best_score = temp_best_score;
        best_sol = temp_best_sol;
    }
}
vector<vector<vector<int>>> Solver_SA::run_SA()
{
    srand(time(nullptr)); // 初始化随机数种子
    cout << "initScore:" << init_score << endl;
    // 重置试探最优解5轮
    for (int iter = 0; iter < 5; ++iter)
    {
        reset();
        init_Temperature();
        annealing();
    }
    // 持续优化15轮
    current_sol = best_sol;
    current_score = best_score;
    for (int iter = 0; iter < 15; ++iter)
    {
        init_Temperature();
        annealing();
    }
    cout << "bestScore:" << best_score << endl;
    return best_sol;
}
extern "C" __declspec(dllexport) int *run_SA(const int *array, const int m, const int n, const int p, const int *pinsCode, const int pinsCodeSize, const int ref_width)
{
    Solver_SA solver = Solver_SA(array, m, n, p, pinsCode, pinsCodeSize, ref_width);
    vector<vector<vector<int>>> best_sol = solver.run_SA();
    int *result = new int[m * n * p];
    // Copy values from the array to the 3D vector
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            for (int k = 0; k < p; ++k)
            {
                result[i * n * p + j * p + k] = best_sol[i][j][k];
            }
        }
    }
    return result;
}
extern "C" __declspec(dllexport) void destroy_array(int *result)
{
    delete[] result;
}
int main()
{
    clock_t start = clock();
    cout << "测试case:SEDFD2,规模39*2,需耗时400s" << endl;
    const int array[] = {39, 1, 13, 8, 32, 155, 61, 1, 29, 8, 1, 120, 35, 1, 13, 2, 31, 125, 36, 1, 13, 2, 31, 125, 41, 1, 9, 24, 4, 200, 59, 1, 21, 24, 10, 150, 60, 1, 21, 24, 10, 145, 40, 1, 32, 5, 1, 155, 42, 1, 7, 5, 1, 170, 43, 1, 7, 5, 1, 170, 56, 1, 10, 5, 1, 120, 0, 1, 30, 5, 0, 120, 0, 1, 30, 5, 0, 120, 65, 1, 17, 28, 1, 170, 49, 1, 5, 21, 1, 170, 51, 1, 15, 21, 1, 170, 52, 1, 15, 21, 1, 170, 0, 1, 5, 21, 0, 135, 0, 1, 5, 21, 0, 135, 0, 1, 5, 21, 0, 135, 44, 1, 12, 27, 1, 120, 47, 1, 24, 26, 1, 165, 48, 1, 24, 26, 1, 165, 53, 1, 9, 26, 12, 120, 57, 1, 21, 26, 27, 150, 58, 1, 21, 26, 27, 145, 45, 1, 27, 9, 1, 150, 46, 1, 27, 9, 1, 145, 37, 1, 26, 20, 1, 165, 38, 1, 26, 20, 1, 165, 54, 1, 25, 16, 1, 145, 55, 1, 25, 16, 1, 140, 62, 1, 11, 16, 1, 145, 63, 1, 11, 16, 1, 140, 33, 1, 31, 29, 1, 125, 34, 1, 31, 29, 1, 125, 32, 1, 28, 19, 1, 170, 64, 1, 4, 11, 17, 170, 50, 1, 4, 13, 25, 170, 1, 0, 3, 8, 0, 220, 30, 0, 29, 8, 0, 220, 2, 0, 13, 2, 3, 155, 0, 0, 13, 2, 31, 125, 3, 0, 21, 24, 27, 130, 4, 0, 21, 24, 27, 130, 15, 0, 9, 24, 23, 120, 5, 0, 7, 5, 0, 140, 6, 0, 7, 5, 0, 135, 7, 0, 7, 5, 0, 140, 8, 0, 7, 5, 0, 135, 24, 0, 22, 5, 0, 155, 31, 0, 30, 5, 0, 120, 9, 0, 14, 28, 0, 140, 10, 0, 15, 21, 0, 140, 11, 0, 15, 21, 0, 135, 12, 0, 15, 21, 0, 140, 13, 0, 15, 21, 0, 135, 18, 0, 5, 21, 0, 140, 19, 0, 5, 21, 0, 135, 14, 0, 23, 27, 0, 120, 16, 0, 24, 26, 0, 140, 17, 0, 9, 26, 18, 190, 25, 0, 21, 26, 30, 120, 0, 0, 21, 26, 27, 145, 0, 0, 21, 26, 27, 145, 20, 0, 27, 9, 0, 140, 0, 0, 27, 9, 1, 145, 21, 0, 26, 20, 0, 140, 0, 0, 26, 20, 1, 165, 22, 0, 18, 16, 14, 140, 26, 0, 11, 16, 0, 160, 0, 0, 11, 16, 1, 140, 0, 0, 11, 16, 1, 140, 23, 0, 13, 29, 22, 155, 0, 0, 31, 29, 1, 125, 27, 0, 28, 19, 0, 140, 28, 0, 6, 11, 0, 160, 29, 0, 18, 13, 6, 140};
    const int pinsCode[] = {8, 16, 20, 19, 2, 15, 7};
    const int m = 2, n = 39, p = 6, pinsCodeSize = 7, ref_width = 50;
    // const int n = arraySize / 12;
    int *result = run_SA(array, m, n, p, pinsCode, pinsCodeSize, ref_width);
    for (int i = 0; i < m * n * p; ++i)
    {
        cout << result[i] << " ";
    }
    cout << endl;
    destroy_array(result);
    clock_t finish = clock();
    cout << "耗时:" << (double)(finish - start) / CLOCKS_PER_SEC << 's' << endl;
    return 0;
}