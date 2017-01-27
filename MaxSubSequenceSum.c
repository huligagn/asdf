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

static int MaxSubSum(const int A[], int Left, int Right)
{
	int MaxLeftSum, MaxRightSum;
	int MaxLeftBoarderSum, MaxRightBoarderSum;
	int LeftBoarderSum, RightBoarderSum;
	int Center, i;

	if(Left == Right)
		if(A[Left] > 0)
			return A[Left];
		else
			return 0;

	Center = (Left + Right) / 2;
	MaxLeftSum = MaxSubSum(A, Left, Center);
	MaxRightSum = MaxSubSum(A, Center, Right);

	MaxLeftBoarderSum = 0; LeftBoarderSum = 0;
	for(i = Center; i >= Left; i--) {
		LeftBoarderSum += A[i];
		if(LeftBoarderSum > MaxLeftBoarderSum)
			MaxLeftBoarderSum = LeftBoarderSum;
	}

	MaxRightBoarderSum = 0; RightBoarderSum = 0;
	for(i = Center; i <= Right; i++) {
		RightBoarderSum += A[i];
		if(RightBoarderSum > MaxRightBoarderSum)
			MaxRightBoarderSum = RightBoarderSum;
	}

	return Max3(MaxLeftSum, MaxRightSum, MaxLeftBoarderSum + MaxRightBoarderSum);
}

int MaxSubSeqenceSum_1(const int A[], int N)
{
	return MaxSubSum(A, 0, N - 1);
}

int MaxSubSequenceSum_2(const int A[], int N)
{
	int ThisSum, MaxSum, j;

	ThisSum = MaxSum = 0;
	for(j = 0; j < N; j++) {
		ThisSum += A[j];

		if(ThisSum > MaxSum)
			MaxSum = ThisSum;
		else if(ThisSum < 0)
			ThisSum = 0;
	}

	return MaxSum;
}

int main(void)
{
	int a[] = {-2, 11, -4, 13, -5, -2};
	int out = MaxSubSeqenceSum_0(a, 6);
	printf(">>>>>>>>>>: %d\n", out);

	return 0;
}