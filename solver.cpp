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
vector<vector<vector<int>>> update_vector(vector<vector<vector<int>>> &s_old, int action)
{
    std::srand(static_cast<unsigned int>(std::time(0)));
    // cout << "Action: " << action << endl;
    // for (const auto &layer : s_old)
    // {
    //     for (const auto &row : layer)
    //     {
    //         for (int value : row)
    //         {
    //             cout << value << " ";
    //         }
    //         cout << endl;
    //     }
    //     cout << endl;
    // }
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
extern "C"
{
    // 创建一个数组
    int *v_compute(int *array, int m, int n, int p, int action)
    {
        // cout << "数组的形状为:m=" << m << " n=" << n << " p=" << p << endl;
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
        vector<vector<vector<int>>> s_vector;
        s_vector.resize(m, std::vector<std::vector<int>>(n, std::vector<int>(p)));
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
        // print_vector(s_vector, m, n, p);
        vector<vector<vector<int>>> new_s_vector = update_vector(s_vector, action);
        // print_vector(new_s_vector, m, n, p);
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
                    s[i * n * p + j * p + k] = new_s_vector[i][j][k];
                }
            }
        }
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