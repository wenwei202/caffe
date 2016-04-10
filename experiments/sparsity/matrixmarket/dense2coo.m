function  dense2coo(filename,ydimen,xdimen)
    [A,rows,cols,entries,rep,field,symm] = mmread(filename);
    mmwrite(sprintf('coo_%s',filename),sparse(A));
    
    [A,rows,cols,entries,rep,field,symm] = mmread(sprintf('coo_%s',filename));
    A = full(A);
    colormap('gray')
    imshow(A==0);
    display('K-mode sparsity:')
    display(block_sparsity(A, ydimen,xdimen));
    display('Elewise sparsity:')
    display(sum(sum(abs(A==0)))/numel(A));
    display('-----------------')
end