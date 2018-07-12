function AAindex_code=AAindex(Seq,aaindex)
% Seq: N*M;
% aaindex:L*21;
% AAIndex_code: N*(M*L);
model=['A'    'R'    'N'    'D'    'C'    'Q'    'E'  ...
    'G'    'H'    'I'    'L'    'K'    'M'    'F' ...  
    'P'    'S'    'T'    'W'    'Y'    'V'   'X'];
[p,q]=size(aaindex);
AAindexModelscale=zeros(p,q);
for a=1:p
    M1=max(aaindex(a,:));
    M2=min(aaindex(a,:));
    for b=1:q
        AAindexModelscale(a,b)=(aaindex(a,b)-M2)/(M1-M2);
    end
end 
AAindexModelscale=AAindexModelscale';
[m,n]=size(Seq);
AAindex_code=[];
for i=1:m
    AAindex_row=[];
    for j=1:n
        for l=1:21 
            if Seq(i,j)==model(l)
                AAindex_ele=AAindexModelscale(l,:); 
                break;
            end
        end
        AAindex_row=[AAindex_row AAindex_ele];
    end
    AAindex_code= [AAindex_code; AAindex_row];
end
end