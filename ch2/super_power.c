long int super_pow(long int X, unsigned int N)
{
	if (N == 0)
		return 1;
	else if (N == 1)
		return X;
	if(IsEven(N))
		return super_pow(X * X, N / 2);
	else
		return super_pow(X * X, N / 2) * X;
}