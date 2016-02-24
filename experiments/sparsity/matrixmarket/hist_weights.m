num = 5;
name = cell(num,1);
figure;
for i=1:num
    name{i} = sprintf('conv%d',i);
    [A,rows,cols,entries,rep,field,symm] = mmread(sprintf('%s.weight',name{i}));
    subplot(1,num,i)
    hist(sum(A'~=0),20)
    xlim([0,Inf])
    xlabel(name{i})
    %title('histogram of nnz of row')
end

