function  Binary_matrix_code= Binary_matrix(seq )
A=eye(21);
B=['A'    'R'    'N'    'D'    'C'    'Q'    'E'  ...
    'G'    'H'    'I'    'L'    'K'    'M'    'F' ...
    'P'    'S'    'T'    'W'    'V'    'Y'   'X'];
[m,n]=size(seq);
matrix_code1=[];  Binary_matrix_code=zeros(m,n-1,21);
for i=1:m
    matrix_code2=[];
    for j=1:(n-1)/2
        for k=1:21
            if seq(i,j)==B(k)
                matrix_code1=A(k,:);
            end
        end
        matrix_code2=[matrix_code2;matrix_code1];
    end
    for j=(n-1)/2+2:n
        for k=1:21
            if seq(i,j)==B(k)
                matrix_code1=A(k,:);
            end
        end
        matrix_code2=[matrix_code2;matrix_code1];
    end
    Binary_matrix_code(i,:,:)=matrix_code2;
end
end