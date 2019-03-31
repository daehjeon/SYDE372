clear all
close all

data = load('lab2_3.mat');
a = data.a;
b = data.b;


j = 1;

while(1)

    while(1)
        random_a = a(randi(size(a,1)),:);
        random_b = b(randi(size(b,1)),:);
        norm_random_a = norm(random_a,2);
        norm_random_b = norm(random_b,2);

        counterA = 1;
        n_a_B = 0;
        n_b_A = 0;
        A_correctlyClassified=[];
        for i=1:size(a,1)
            sample = a(i,:);
            G = MED(random_a,random_b,sample);
            if norm_random_a>norm_random_b && G<0
                n_a_B=n_a_B+1;
            elseif norm_random_a<norm_random_b && G>0
                n_a_B=n_a_B+1;
            else
                A_correctlyClassified(counterA) = i;
                counterA=counterA+1;
            end
        end

        counterB = 1;
        B_correctlyClassified=[];
        for i=1:size(b,1)
            sample = b(i,:);
            G = MED(random_a,random_b,sample);
            if norm_random_b>norm_random_a && G<0
                n_b_A=n_b_A+1;
            elseif norm_random_b<norm_random_a && G>0
                n_b_A=n_b_A+1;
            else
                B_correctlyClassified(counterB) = i;
                counterB=counterB+1;
            end
        end
        if n_b_A==0 || n_a_B==0
            G_j(j).a = random_a;
            G_j(j).b = random_b;
            n_b_A_j(j) = n_b_A;
            n_a_B_j(j) = n_a_B;
            break
        end
    end
    j = j+1;
    if n_a_B==0
        b(B_correctlyClassified,:)=[];
    end
    if n_b_A==0
        a(A_correctlyClassified,:)=[];
    end
    if isempty(b) || isempty(a)
        break
    end
end
    
%%====================================== Test step

% for a given sample x_test
a = data.a;
b = data.b;

errorA=0;
for i=1:size(a,1)
    x_test = a(i,:);
    prediction = getTestError(G_j,n_b_A_j,n_a_B_j,x_test);
    if prediction~='A'
        errorA=errorA+1;
    end
end
errorB=0;
for i=1:size(b,1)
    x_test = b(i,:);
    prediction = getTestError(G_j,n_b_A_j,n_a_B_j,x_test);
    if prediction~='B'
        errorB=errorB+1;
    end
end

disp()



















