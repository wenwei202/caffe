function sparsity = block_sparsity(mat, ydimen,xdimen)
    n = size(mat,1);
    c = size(mat,2);
    blk_num_n = ceil(n/ydimen);
    blk_num_c = ceil(c/xdimen);
    count = 0;
    for blk_n = 0:blk_num_n-1
        for blk_c = 0:blk_num_c-1
            n_range = blk_n*ydimen+1:blk_n*ydimen+ydimen;
            if blk_n*ydimen+ydimen>n
                n_range = blk_n*ydimen+1:n;
            end
            c_range = blk_c*xdimen+1:blk_c*xdimen+xdimen;
            if blk_c*xdimen+xdimen>c
                c_range = blk_c*xdimen+1:c;
            end
            submat = mat(n_range,c_range);
            count = count + (sum(sum(abs(submat)))==0);
        end
    end
    sparsity = count/(blk_num_n*blk_num_c);
end