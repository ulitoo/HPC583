#include <iostream>
#include <vector>

using namespace std;

vector<int> longMultiplication(const vector<int>& num1, const vector<int>& num2) {
    int m = num1.size();
    int n = num2.size();
    vector<int> result(m + n, 0);

    for (int i = m - 1; i >= 0; i--) {
        for (int j = n - 1; j >= 0; j--) {
            int product = (num1[i]) * (num2[j]);
            int sum = product + result[i + j + 1];

            result[i + j] += sum / 10;
            result[i + j + 1] = sum % 10;

            // Print calculation steps for each pair of digits
            cout << "Multiply " << num1[i] << " and " << num2[j] << ": " << num1[i] << " * " << num2[j] << " = " << product << endl;
            cout << "Add " << product << " to result[" << i + j + 1 << "]: " << result[i + j + 1] << " + " << product << " = " << sum << endl;
            cout << "Result after step: ";
            for (int digit : result) {
                cout << digit;
            }
            cout << endl;
        }
    }

    // Remove leading zeros if any
    while (result.size() > 1 && result[0] == 0) {
        result.erase(result.begin());
    }

    return result;
}

int main() {
    string num1, num2;
    cout << "Enter the first number: ";
    cin >> num1;
    cout << "Enter the second number: ";
    cin >> num2;

    vector<int> num1Vec, num2Vec;

    for (char c : num1) {
        num1Vec.push_back(c - '0');
    }

    for (char c : num2) {
        num2Vec.push_back(c - '0');
    }

    vector<int> result = longMultiplication(num1Vec, num2Vec);

    cout << "Final Result: ";
    for (int digit : result) {
        cout << digit;
    }
    cout << endl;

    return 0;
}
