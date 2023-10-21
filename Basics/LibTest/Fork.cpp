#include <iostream>
#include <unistd.h>
#include <string>
#include <sys/wait.h>

using namespace std;

int main()
{
    pid_t pid;
    int status;

    pid = fork();

    if (pid == 0)
    {
        cout << "Child process number "<<pid<<"\n" << endl;
        const char *args[] = {"ls", nullptr};
        execvp(args[0], (char **)args);
        cout << "NOT PRINTED" << endl;
    }
    else if (pid>0)
    {
        cout << "\nParent process number "<<pid<<"\n" << endl;
        wait(&status);
        cout << "\nChild process exit with status "<< status << "." << endl;
    }
    else
    {
        cout << "Error Fork failed" << endl;
        return 1;
    }

    return 0;
}