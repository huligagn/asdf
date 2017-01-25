#include <stdio.h>

// version_0, O(N^2)
int MaxSubSeqenceSum_0(const int A[], int N)
{
	int i, j, sum, max;
	max = 0;
	for (i = 0; i < N; i++) {
		sum = 0;
		for (j = i; j < N; j++) {
			sum += A[j];
			if (sum > max)
				max = sum;
		}
	}

	return max;
}

int main(void)
{
	int a[] = {-2, 11, -4, 13, -5, -2};
	int out = MaxSubSeqenceSum_0(a, 6);
	printf(">>>>>>>>>>: %d\n", out);

	return 0;
}