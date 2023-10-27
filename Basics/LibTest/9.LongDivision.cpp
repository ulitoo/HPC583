
#include <iostream>
#include <vector>
#include <string>

using namespace std;

int main()
{

    int number, divisor;

    cout << "This is a Long Division Demo \n Get The first number (Dividend): ";
    cin >> number;
    cout << "\nNow get the Divisor number: ";
    cin >> divisor;
    std::string numbers = std::to_string(number);
    std::string divisors = std::to_string(divisor);
    int int_numbers[size(numbers)], int_divisors[size(divisors)];

    // printf("The first number is %i \n",number);

    for (int i = 0; i < size(numbers); i++)
    {
        int_numbers[i] = numbers[i] - '0';
        printf("Dividend %i:\t%i\n", i, int_numbers[i]);
    }
    cout<<"\n";
    for (int i = 0; i < size(divisors); i++)
    {
        int_divisors[i] = divisors[i] - '0';
        printf("Divisor %i:\t%i\n", i, int_divisors[i]);
    }

    return 0;
}
