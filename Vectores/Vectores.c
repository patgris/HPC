#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 100000

int suma(int a, int b)
{
	int *result;
	result = (int*) malloc (sizeof(int));
	*result=a+b;
	return *result;
}

int main()
{
	int *v1, *v2, *v3, *i;
	v1 = (int*) malloc (SIZE*sizeof(int));
	v2 = (int*) malloc (SIZE*sizeof(int));
	v3 = (int*) malloc (SIZE*sizeof(int));
	i = (int*) malloc (sizeof(int));
	*i=0;
	clock_t start = clock();
	for (*i=0; *i<SIZE; (*i)++)
	{
		v1[*i]=(*i)+2;
		v2[*i]=(*i)+3;
		v3[*i] = suma(v1[*i], v2[*i]);
		printf("%d\n", v3[*i]);
	}
	printf("Tiempo transcurrido: %f", ((double)clock() - start) / CLOCKS_PER_SEC);
	return 0;
}
