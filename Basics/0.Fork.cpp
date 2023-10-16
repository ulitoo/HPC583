#include <iostream>
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>

int main()
{
    int status;
    char *command = "ls";
    char *argument_list[] = {"ls", NULL};

    printf("Before calling execvp()\n");

    printf("Creating another process using fork()...\n");

    int cpujbg;
    cpujbg=sched_getcpu();  std::cout<<"cpu: "<<(cpujbg)<<"\n";

    if (fork() == 0)
    {
        // Newly spawned child Process. This will be taken over by "ls -l"
        cpujbg=sched_getcpu();  std::cout<<"Ccpu: "<<(cpujbg)<<"\n";
        int status_code = execvp(command, argument_list);

        printf("ls -l has taken control of this child process. This won't execute unless it terminates abnormally!\n");

        if (status_code == -1)
        {
            printf("Terminated Incorrectly\n");
            return 1;
        }
    }
    else
    {
        cpujbg=sched_getcpu();  std::cout<<"Pcpu: "<<(cpujbg)<<"\n";
        wait(&status);
        // Old Parent process. The C program will come here
        printf("This line will be printed\n");
        cpujbg=sched_getcpu();  std::cout<<"P2cpu: "<<(cpujbg)<<"\n";
        std::cout<<(status)<<"\n";
    }

    return 0;
}