
#include <iostream>
#include <vector>
#include <cmath>
#include <string>

using namespace std;

//Experiment with other types
// long double
// double
// float
/*
int main()
{
    long double one=1.;
    long double half=1./2.;
    long double tmp;
    int j;

    for (j=0;;j++){
        tmp = 1.+pow(half,static_cast<long double>(j)); 
        
        if ((tmp-one) == 0.){
            break;
        } 

    }
    
    std::cout << "Iter: " << j-1 << " Value:" << pow(half,static_cast<long double>(j-1)) << endl;

    return 0;
}
*/
int main()
{
     double one=1.;
     double half=1./2.;
     double tmp;
    int j;

    for (j=0;;j++){
        tmp = 1.+pow(half,static_cast< double>(j)); 
        
        if ((tmp-one) == 0.){
            break;
        } 

    }
    
    std::cout << "Iter: " << j-1 << " Value:" << pow(half,static_cast< double>(j-1)) << endl;

    return 0;
}