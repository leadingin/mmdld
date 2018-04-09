function [acc,acc_ite,A] = MyBDA1(X_src,Y_src,X_tar,Y_tar,options,src,tgt)
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
    origin = [originacc,svmacc];

    X = [X_src',X_tar'];
	X = X*diag(sparse(1./sqrt(sum(X.^2))));
	[m,n] = size(X);
	ns = size(X_src,1);
	nt = size(X_tar,1);
	e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
	C = length(unique(Y_src));

    %% Centering matrix H
	H = eye(n) - 1/n * ones(n,n);
	%%% M0
	M = e * e' * C;  %multiply C for better normalization

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
        end
   
        %normalization for better classification performance
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
        knn_model = fitcknn(Zs,Y_src,'NumNeighbors',1);
        Y_tar_pseudo1 = knn_model.predict(Zt);
        acc = length(find(Y_tar_pseudo1==Y_tar))/length(Y_tar); 
        %save(['/media/data/ld/BDA/dataget/' src tgt 'source.mat'],'Zs','acc');
	    %save(['/media/data/ld/BDA/dataget/' src tgt 'target.mat'],'Zt','acc');
        fprintf('Iteration [%2d]:BDA+NN=%0.4f\n',i,acc);
        Y_tar_pseudo=Y_tar_pseudo1;
        acc_ite = [acc_ite;acc];
        svmac1 = [svmac1;svmacc1];
        svmac2 = [svmac2;svmacc2];
        svmac3 = [svmac3;svmacc3];
	end
    %save(['/media/data/ld/BDA2019fts/data2to1/' src tgt 's.mat'],'Zs','acc','acc_ite','svmac1','svmac2','svmac3','origin');
    %save(['/media/data/ld/BDA2019fts/data2to1/' src tgt 't.mat'],'Zt','acc','acc_ite','svmac1','svmac2','svmac3','origin');
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
%
% Gustavo Camps-Valls
% 2006(c)
% Jordi (jordi@uv.es), 2007
% 2007-11: if/then -> switch, and fixed RBF kernel
% Modified by Mingsheng Long
% 2013(c)
% Mingsheng Long (longmingsheng@gmail.com), 2013

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
