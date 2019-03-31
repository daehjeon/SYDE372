function prediction = getTestError(G_j,n_b_A_j,n_a_B_j,x_test)


    for j=1:length(G_j)
        G = MED(G_j(j).a,G_j(j).b,x_test);
        norm_random_a = norm(G_j(j).a,2);
        norm_random_b = norm(G_j(j).b,2);
        
        if norm_random_a>norm_random_b && G>0 || norm_random_a<norm_random_b && G<0
            if n_b_A_j(j)==0
                prediction='A';
                break;
            end
        elseif norm_random_a<norm_random_b && G>0 || norm_random_a>norm_random_b && G<0
            if n_a_B_j(j)==0
                prediction='B';
                break;
            end
        end        
    end%End of For
end