function PseAAC_seq=PseAAC1(seqs)
[m,n]=size(seqs);
lamda=20;w=0.05;
amino=['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','X'];
H1=[0.62	0.29	-0.9	-0.74	1.19	0.48	-0.4	1.38	-1.5	1.06	0.64	-0.78	0.12	-0.85	-2.53	-0.18	-0.05	1.08	0.81	0.26];
H2=[-0.5	-1	3	3	-2.5	0	-0.5	-1.8	3	-1.8	-1.3	0.2	0	0.2	3	0.3	-0.4	-1.5	-3.4	-2.3];
M=[15	47	59	73	91	1	82	57	73	57	75	58	42	72	101	31	45	43	130	107];
% Standard conversion
h11=sum(H1)/20;h21=sum(H2)/20;m1=sum(M)/20;
h12=0;h22=0;m2=0;
for i=1:20
    h12=h12+(H1(i)-h11)^2;
    h22=h22+(H2(i)-h21)^2;
    m2=m2+(M(i)-m1)^2;
end
for j=1:20
    H1(j)=(H1(j)-h11)/sqrt(h12/20);
    H2(j)=(H2(i)-h21)/sqrt(h22/20);
    M(j)=(M(j)-m1)/sqrt(m2/20);
end
H1=[H1 0];H2=[H2 0];M=[M 0];
PseAAC_seq=zeros(m,20+lamda);
for i=1:m
    seq_singal = seqs(i,:); 
    for j=1:20
        aa=amino(j);
        num=length(strfind(seq_singal,aa));   % find the letter
        PseAAC_seq(i,j)=num/n;
    end
    for space=0:lamda-1
        delta=zeros(n-space-1,1);
        for k=1:n-space-1
%             a1=find(amino==seq_singal(k));
%             a2=find(amino==seqs(i,k+space+1));
            if ismember(seqs(i,k),amino)
                a1=find(amino==seqs(i,k));
            else
                a1=21;
            end
            if ismember(seqs(i,k+space+1),amino)
                a2=find(amino==seqs(i,k+space+1));
            else
                a2=21;
            end
            delta(k)=((H1(a1)-H1(a2)).^2+(H2(a1)-H2(a2)).^2+(M(a1)-M(a2)).^2)/3;
        end
        PseAAC_seq(i,j+space+1)=sum(delta)/(n-space-1);
    end
end

fenmu=zeros(m,1);
for i=1:m
    sum1=sum(PseAAC_seq(i,1:20));
    sum2=w*sum(PseAAC_seq(i,21:20+lamda));
    fenmu(i)=sum1+sum2;
end

for i=1:m
    for j=1:20
        PseAAC_seq(i,j)=PseAAC_seq(i,j)/fenmu(i);
    end
    for j=21:20+lamda
        PseAAC_seq(i,j)=w*PseAAC_seq(i,j)/fenmu(i);
    end
end