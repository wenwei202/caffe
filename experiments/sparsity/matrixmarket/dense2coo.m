function  dense2coo(filename)
    [A,rows,cols,entries,rep,field,symm] = mmread(filename);
    %comment='%MatrixMarket matrix coordinate real general';
   mmwrite(sprintf('coo_%s',filename),sparse(A));
end