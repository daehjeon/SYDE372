clear all
close all

data = load('lab2_1.mat');
a = data.a;
b = data.b;
% 
% mu_hat_a = sum(a)/length(a);
% 
% % sum = 0;
% % for i=1:length(a)
% %     sum = sum + (a(i) - mu_hat)^2;
% % end
% % v_hat = sum/length(a);
% 
% v_hat_a = sum ( (a-mu_hat_a).^2 )/length(a);
% 
% p_hat_a = mvnpdf(a',mu_hat_a,v_hat_a);
% p_a = mvnpdf(a',5,1);
% 
% figure
% plot(a,p_hat_a,'b*')
% hold on
% plot(a,p_a,'rs')
% title('Data a')
% %===============================
% mu_hat_b = sum(b)/length(b);
% v_hat_b = sum ( (b-mu_hat_b).^2 )/length(b);
% 
% p_hat_b = mvnpdf(b',mu_hat_b,v_hat_b);
% p_b = 1.*exp(-b);
% 
% figure
% plot(b,p_hat_b,'b*')
% hold on
% plot(b,p_b,'rs')
% title('Data b')
% %=====================================Part1-2
% lambda_hat_a = length(a)./sum(a);
% p_hat_a = lambda_hat_a.*exp(-lambda_hat_a.*a);
% p_a = mvnpdf(a',5,1);
% 
% figure
% plot(a,p_hat_a,'b*')
% hold on
% plot(a,p_a,'rs')
% title('Data a')
% %===============================
% lambda_hat_b = length(b)./sum(b);
% p_hat_b = lambda_hat_b.*exp(-lambda_hat_b.*b);
% p_b = 1.*exp(-b);
% 
% figure
% plot(b,p_hat_b,'b*')
% hold on
% plot(b,p_b,'rs')
% title('Data b')

%====================================================
h1 = 0.1;
h2 = 0.4;
p_a = mvnpdf(a',5,1);
for i=1:length(a)
    parzen_hat_a(i) = parzenF(a,a(i),h1);
end
figure
plot(a,parzen_hat_a,'b*')
hold on
plot(a,p_a,'rs')
title('Data a_ Parzen')
%--------------------
for i=1:length(a)
    parzen2_hat_a(i) = parzenF(a,a(i),h2);
end
figure
plot(a,parzen2_hat_a,'b*')
hold on
plot(a,p_a,'rs')
title('Data a_ Parzen')







