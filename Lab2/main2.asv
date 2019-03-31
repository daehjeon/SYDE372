clear all
close all

data = load('lab2_2.mat');
al = data.al;
bl = data.bl;
cl = data.cl;
%
mu_al = sum(al)./size(al,1);
mu_bl = sum(bl)./size(bl,1);
mu_cl = sum(cl)./size(cl,1);
% duplicated_mu_al = repmat(mu_hat_al,length(al),1);
cov_al = get_cov(al,mu_al);
cov_bl = get_cov(bl,mu_bl);
cov_cl = get_cov(cl,mu_cl);

% type = 1;
% plotML(al,bl,cl, mu_al, cov_al, mu_bl, cov_bl, mu_cl, cov_cl,type)

type = 2;
plotML(al,bl,cl, mu_al, cov_al, mu_bl, cov_bl, mu_cl, cov_cl,type)