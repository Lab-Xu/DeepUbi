function vocab_code=Vocab(seqs)
amino =['A'    'C'    'D'    'E'    'F'    'G'    'H'  'I'    'K'  'L'    'M'    'N'    'P'    'Q'   'R'    'S'    'T'    'V'    'W'    'Y'   'X' ];
% matrix_code=zeros(length(amino),length(amino),size(seqs,1));
[m,n]=size(seqs);
vocab_code=zeros(m,n);
for i=1:m
    seq_singal = seqs(i,:); 
    for j=1:n
        if ismember(seq_singal(j),amino)
            num=find(amino==seq_singal(j));
        else
            num=21;
        end   % find the letter
        vocab_code(i,j)=num-1;
    end
end
vocab_code(:,(n+1)/2)=[];
end
