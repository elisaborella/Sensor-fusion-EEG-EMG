function [myZC, myZCx, myZCy, ys] = myZC(x, y)
    ys = sign(y);
    ysd = diff(ys);
    myZC = find(ysd ~= 0);

    % Error management
    if isempty(myZC)
        myZC = nan;
        myZCx = nan;
        myZCy = nan;
        ys = nan;
        return
    end

    % Initialize myZCx and myZCy with NaNs
    myZCx = nan(size(myZC));
    myZCy = nan(size(myZC));
    
    % Refine x-position
    for zc = 1:length(myZC)
        clear xos yos zcid id

        % Over-sample x,y between two consecutive sample including a zero-cross
        xos = linspace(x(myZC(zc)), x(myZC(zc) + 1), 1000); % linspace(X1, X2) generates a row vector of 1000 linearly equally spaced points between X1 and X2
        yos = linspace(y(myZC(zc)), y(myZC(zc) + 1), 1000);

        % Find indices where yos is close to zero
        zcid = find(abs(yos) <= 0.015);

        % Skip if zcid is empty
        if isempty(zcid)
            continue;
        end

        % Use the closest point to zero
        [~, id] = min(abs(yos(zcid)));

        % Store refined zero-crossing points
        myZCx(zc) = xos(zcid(id));
        myZCy(zc) = yos(zcid(id));
    end

end
