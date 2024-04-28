import numpy as np
from tqdm import tqdm

def DIGing_mgrad(x_0, W, iterations, A_list, b_list, A, b, alpha, grad, grad_full, K):
    
    temp1 = grad_full(A_list, b_list, x_0)

    y = temp1

    norms = np.zeros(iterations)
    
    for i in tqdm(range(iterations)):
        
        x_1 = np.dot(W, x_0) - alpha*y

        temp2 = grad_full(A_list, b_list, x_1)
                
        y = np.dot(W, y) + temp2 - temp1
        
        x_0 = x_1

        temp1 = temp2
        
        norms[i] = np.linalg.norm(np.mean(grad_full(A_list, b_list, np.tile(np.mean(x_1, 0).T, (np.shape(W)[0], 1))), 0))        

        # norms[i] = np.linalg.norm(grad(A, b, np.mean(x_1, 0).T))        


        for k in range(K - 1):
            
            x_1 = x_0 - alpha*y

            temp2 = grad_full(A_list, b_list, x_1)
            
            y = y + temp2 - temp1
            
            x_0 = x_1

            temp1 = temp2
            
    return norms

def DIGing_mgrad_track(x_0, W, iterations, A_list, b_list, A, b, alpha, grad, grad_full, K, x_opt):
        
    temp1 = grad_full(A_list, b_list, x_0)

    y = temp1
    # norms = np.zeros(iterations)
    opt_error = np.zeros(iterations)
    con_x_error = np.zeros(iterations)
    # con_y_error = np.zeros(iterations)
    # f = np.zeros(iterations)
    
    for i in tqdm(range(iterations)):
        
        x_1 = np.dot(W, x_0) - alpha*y
        
        # norms[i] = np.linalg.norm(grad(A, b, np.mean(x_1, 0)[np.newaxis].T))
        opt_error[i] = np.linalg.norm(np.mean(x_1, 0)[np.newaxis].T - x_opt)
        # f[i] = f_value(A, b, np.mean(x_1, 0)[np.newaxis].T)

        temp2 = grad_full(A_list, b_list, x_1)

        con_x_error[i] = np.linalg.norm(np.mean(x_1, 0) - x_1)
        
        y = np.dot(W, y) + temp2 - temp1

        # con_y_error[i] = np.linalg.norm(np.mean(y, 0) - y)
        
        x_0 = x_1

        temp1 = temp2
        
        # norms[i] = np.linalg.norm(grad(A, b, np.mean(x_1, 0)[np.newaxis].T))
        
        for k in range(K - 1):
            
            x_1 = x_0 - alpha*y

            temp2 = grad_full(A_list, b_list, x_1)
            
            y = y + temp2 - temp1
            
            x_0 = x_1

            temp1 = temp2
            
    # return norms, opt_error, con_x_error, con_y_error, f
    return opt_error, con_x_error