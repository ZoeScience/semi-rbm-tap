function A = SortFilter(A)
	[B,ind]=sort(A);
	count=numel(A);
	topNumber=0.1 * count;
	for i = 1 : topNumber
		index=B[i];
		A[index]=0;
	end
end