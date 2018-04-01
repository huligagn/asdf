unsigned int gcd(unsigned int M, unsigned int N)
{
	unsigned int rem;

	while(M != N) {
		rem = M % N;
		M = N;
		N = rem;
	}

	return M;
}