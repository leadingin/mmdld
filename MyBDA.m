function [acc,acc_ite,A] = MyBDA(X_src,Y_src,X_tar,Y_tar,options,src,tgt)
    % Inputs:
    %%% X_src  :source feature matrix, ns * m
    %%% Y_src  :source label vector, ns * 1
    %%% X_tar  :target feature matrix, nt * m
    %%% Y_tar  :target label vector, nt * 1
    %%% options:option struct
    % Outputs:
    %%% acc    :final accuracy using knn, float
    %%% acc_ite:list of all accuracies during iterations
    %%% A      :final adaptation matrix, (ns + nt) * (ns + nt)

    % Reference:
    % Jindong Wang, Yiqiang Chen, Shuji Hao, and Zhiqi Shen.
    % Balanced distribution adaptation for transfer learning.
    % ICDM 2017.

	%% Set options
	lambda = options.lambda;              %% lambda for the regularization
	dim = options.dim;                    %% dim is the dimension after adaptation, dim <= m
	kernel_type = options.kernel_type;    %% kernel_type is the kernel name, primal|linear|rbf
	gamma = options.gamma;                %% gamma is the bandwidth of rbf kernel
	T = options.T;                        %% iteration number
    mu = options.mu;                      %% balance factor \mu
    mode = options.mode;                  %% 'BDA' or 'W-BDA'

    k_model = fitcknn(X_src,Y_src,'NumNeighbors',1);
    Y_get = k_model.predict(X_tar);
    originacc = length(find(Y_get==Y_tar))/length(Y_tar);
    Y_src=double(Y_src);
    Y_tar=double(Y_tar);
    X_src=double(X_src);
    X_tar=double(X_tar);
    svmmodel = svmtrain(Y_src,X_src,'-t 0 -q');
    predict_label = svmpredict(Y_tar, X_tar,svmmodel);
    svmacc = length(find(predict_label==Y_tar))/length(Y_tar);
    fprintf('originacc:%.4f,svmacc:%.4f',originacc,svmacc);
    distcond=0.0;
    for j = 1 : 10
        Xs0=X_src(Y_src==j);
        Xt0=X_tar(Y_tar==j);
        distper0=mmd(Xs0,Xt0,gamma);
        distcond=distcond+distper0;
    end
    X = [X_src',X_tar'];
    %distall1=mmd(X_src,X_tar,gamma)
    %KK = kernel_bda(kernel_type,X,[],gamma);
    mean2=mean(X,1);
    mean22=mean2(1);

	X = X*diag(sparse(1./sqrt(sum(X.^2))));
    XXX=X';
    %k_model = fitcknn(X(:,1:3394)',Y_src,'NumNeighbors',1);
    %Y_get = k_model.predict(X(:,3395:end)');
    %originacc = length(find(Y_get==Y_tar))/length(Y_tar);
    %fprintf('21originacc:%.4f',originacc);
    %mean3=mean(X,2);
    %mean4=mean(X,1);
    %mean33=mean3(1)
    %mean44=mean4(1)
    %KKK = kernel_bda(kernel_type,X,[],gamma);
	[m,n] = size(X);
	ns = size(X_src,1);
	nt = size(X_tar,1);
	e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
	C = length(unique(Y_src));

    %% Centering matrix H
	H = eye(n) - 1/n * ones(n,n);
	%%% M0
	M = e * e' * C;  %multiply C for better normalization
    %M=e*e';
    KK = kernel_bda(kernel_type,X,[],gamma);
    distall1=sqrt(trace(KK*M));
    distall11=mmd(X_src,X_tar,gamma);
    distall12=mmd(XXX(1:ns,:),XXX(ns+1:end,:),gamma)
    dist=distall11+distcond;
    fprintf('distall11=[%0.4f],dist=%0.4f\n',distall11,dist);
    origin = [distall12,distall11,dist,originacc,svmacc];
    %distall2=sqrt(trace(KKK*MM))


    acc_ite = [];
    svmac1 = [];
    svmac2 = [];
    svmac3 = [];
	Y_tar_pseudo = [];

	%% Iteration
	for i = 1 : T
        %%% Mc
        N = 0;
        if ~isempty(Y_tar_pseudo) && length(Y_tar_pseudo)==nt
            for c = reshape(unique(Y_src),1,C)
                e = zeros(n,1);
                if strcmp(mode,'W-BDA')
                    Ns = length(Y_src(Y_src==c,:));
                    Nt = length(Y_tar_pseudo(Y_tar_pseudo == c,:));
                    Ps = Ns / length(Y_src);
                    Pt = Nt / length(Y_tar_pseudo);
                else
                    Ps = 1;
                    Pt = 1;
                end
                e(Y_src==c) = sqrt(Ps) / length(find(Y_src==c));
                e(ns+find(Y_tar_pseudo==c)) = -sqrt(Pt) / length(find(Y_tar_pseudo==c));
                e(isinf(e)) = 0;
                N = N + e*e';
            end
        end
        if mu == 1
            mu = 0.999;
        end
        M = (1 - mu) * M + mu * N;
        M = M / norm(M,'fro');

        %% Calculation
        if strcmp(kernel_type,'primal')
            [A,~] = eigs(X*M*X'+lambda*eye(m),X*H*X',dim,'SM');
            Z = A'*X;
        else
            K = kernel_bda(kernel_type,X,[],gamma);
            [A,~] = eigs(K*M*K'+lambda*eye(n),K*H*K',dim,'SM');
            Z = A'*K;
            dis1=sqrt(trace(K*M*K'));
        end

        %normalization for better classification performance
        %ZZ = Z ./ repmat(sum(Z,1),size(Z,1),1);
		%ZZ = zscore(ZZ,1,2);
        Z = Z*diag(sparse(1./sqrt(sum(Z.^2))));
        Zs = Z(:,1:size(X_src,1))';
        Zt = Z(:,size(X_src,1)+1:end)';
        svmmodel1 = svmtrain(Y_src,Zs,'-t 0 -q');
        predict_label1 = svmpredict(Y_tar, Zt,svmmodel1);
        svmmodel2 = svmtrain(Y_src,Zs,'-t 2 -q');
        predict_label2 = svmpredict(Y_tar, Zt,svmmodel2);
        svmmodel3 = svmtrain(Y_src,Zs,'-t 0 -c 0.1 -q');
        predict_label3 = svmpredict(Y_tar, Zt,svmmodel3);
        svmacc1 = length(find(predict_label1==Y_tar))/length(Y_tar);
        svmacc2 = length(find(predict_label2==Y_tar))/length(Y_tar);
        svmacc3 = length(find(predict_label3==Y_tar))/length(Y_tar);
        distcond1=0.0;
        for k = 1 : 10
            Xs1=Zs(Y_src==k);
            Xt1=Zt(Y_tar==k);
            distper1=mmd(Xs1,Xt1,gamma);
            distcond1=distcond1+distper1;
        end

        %KLD = kernel_bda(kernel_type,Z,[],gamma);
        %distLD = sqrt(trace(KLD*M));
        distLD1=mmd(Zs,Zt,gamma);
        dist11=distLD1 + distcond1;
        fprintf('distLD1=[%0.4f],dist11=%0.4f\n',distLD1,dist11);
        knn_model = fitcknn(Zs,Y_src,'NumNeighbors',1);
        Y_tar_pseudo1 = knn_model.predict(Zt);
        acc = length(find(Y_tar_pseudo1==Y_tar))/length(Y_tar);
        %save(['/media/data/ld/BDA/dataget/' src tgt 'source.mat'],'Zs','acc');
	    %save(['/media/data/ld/BDA/dataget/' src tgt 'target.mat'],'Zt','acc');
        fprintf('Iteration [%2d]:BDA+NN=%0.4f\n',i,acc);
        Y_tar_pseudo=predict_label1;
        acc_ite = [acc_ite;acc];
        svmac1 = [svmac1;svmacc1];
        svmac2 = [svmac2;svmacc2];
        svmac3 = [svmac3;svmacc3];
	end
    now=[distLD1,dist11];
    save(['/media/data/ld/BDA2019fts/data1to1/' src tgt 's.mat'],'Zs','acc','acc_ite','svmac1','svmac2','svmac3','origin','now');
    save(['/media/data/ld/BDA2019fts/data1to1/' src tgt 't.mat'],'Zt','acc','acc_ite','svmac1','svmac2','svmac3','origin','now');
end

% With Fast Computation of the RBF kernel matrix
% To speed up the computation, we exploit a decomposition of the Euclidean distance (norm)
%
% Inputs:
%       ker:    'linear','rbf','sam'
%       X:      data matrix (features * samples)
%       gamma:  bandwidth of the RBF/SAM kernel
% Output:
%       K: kernel matrix

function K = kernel_bda(ker,X,X2,gamma)

    switch ker
        case 'linear'

            if isempty(X2)
                K = X'*X;
            else
                K = X'*X2;
            end

        case 'rbf'

            n1sq = sum(X.^2,1);
            n1 = size(X,2);

            if isempty(X2)
                D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*X'*X;
            else
                n2sq = sum(X2.^2,1);
                n2 = size(X2,2);
                D = (ones(n2,1)*n1sq)' + ones(n1,1)*n2sq -2*X'*X2;
            end
            K = exp(-gamma*D); 

        case 'sam'

            if isempty(X2)
                D = X'*X;
            else
                D = X'*X2;
            end
            K = exp(-gamma*acos(D).^2);

        otherwise
            error(['Unsupported kernel ' ker])
    end
end
function [ dist ] = mmd( sourcefeatures,targetfeatures,sigma )
    Kss = rbf_dot(sourcefeatures,sourcefeatures,sigma);
    Kst = rbf_dot(sourcefeatures,targetfeatures,sigma);
    Kts = rbf_dot(targetfeatures,sourcefeatures,sigma);
    Ktt = rbf_dot(targetfeatures,targetfeatures,sigma);
    K = [[Kss,Kst];[Kts,Ktt]];
    n1 = size(sourcefeatures,1);
    n2 = size(targetfeatures,1);
    L = zeros(n1+n2);
    L(1:n1,1:n1) = 1/(n1^2);
    L(n1+1:end,n1+1:end) = 1/(n2^2);
    L(n1+1:end,1:n1) = -1/(n1*n2);
    L(1:n1,n1+1:end) = -1/(n1*n2);
    dist = sqrt(trace(K*L));
end
function [H]=rbf_dot(patterns1,patterns2,sigma)
    size1=size(patterns1);
    size2=size(patterns2);
    G = sum((patterns1.*patterns1),2);
    H = sum((patterns2.*patterns2),2);
    Q = repmat(G,1,size2(1));
    R = repmat(H',size1(1),1);
    H = Q + R - 2*patterns1*patterns2';
    H=exp(-H*sigma);
end

