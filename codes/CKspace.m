function CKspace_seq=CKspace(seqs)
space_n=5; 
amino =['A'    'C'    'D'    'E'    'F'    'G'    'H'  'I'    'K'  'L'    'M'    'N'    'P'    'Q'   'R'    'S'    'T'    'V'    'W'    'Y'   'X' ];
% matrix_code=zeros(length(amino),length(amino),size(seqs,1));
[m,n]=size(seqs);
M=zeros(m,length(amino)*length(amino),space_n);
for space=0:space_n-1
    matrix_code=zeros(length(amino),length(amino),m);
    for j = 1:m
        seq_singal = seqs(j,:);  
        for i=1:n-space-1  
            a1=find(amino==seq_singal(i));
            a2=find(amino==seq_singal(i+space+1));
            matrix_code(a1,a2,j)=matrix_code(a1,a2,j)+1/(n-space-1);
        end
    end
    for k=1:m
        sub_code(k,:) = reshape(matrix_code(:,:,k)',1,length(amino)*length(amino));  
    end
    M(:,:,space+1)=sub_code(:,:);
end
CKspace_seq=zeros(m,length(amino)*length(amino)*space_n);
for i=1:m
    temp=[];
    for j=1:space_n
        temp=[temp M(i,:,j)];
    end
    CKspace_seq(i,:)=temp;
end