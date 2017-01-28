#include <stdio.h>

int Binary_search(const int A[], int X, int N)
{
	int Low, Mid, High;

	Low = 0;
	High = N - 1;

	while(Low <= High) {
		Mid = (Low + High) / 2;
		if(X < A[Mid])
			High = Mid - 1;
		else if(X > A[Mid])
			Low = Mid + 1;
		else
			return Mid;
	}

	return -1; // -1 represents notfound.
}

int main(void)
{
	int foo[] = {1,2,3,4,5,6,7,8,9};
	int target, index;
	target = 10;
	index = Binary_search(foo, target, 9);
	printf("the index of %d in the array: %d\n", target, index);

	return 0;
}
