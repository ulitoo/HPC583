
#include <iostream>
#include <vector>
#include <string>

using namespace std;

int main()
{
    int iteration = 5;
    vector<string> msg{"Hello", "C++", "World", "from", "VS Code", "and the C++ extension!!!"};
    for (const string &word : msg)
    {
        cout << word << " "; // << iteration*iteration <<"  ";
        iteration++;
    }
    cout << endl;

    return 0;
}
