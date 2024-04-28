import numpy as np
from tqdm import tqdm

def DIGing_R(x_0, W, iterations, A_list, b_list, A, b, alpha, grad, grad_full, p, seed):

    np.random.seed(seed)
    
    temp1 = grad_full(A_list, b_list, x_0)

    y = temp1

    norms = np.zeros(iterations)
    
    for i in tqdm(range(iterations)):

        if np.random.rand() <= p:
        
            x_1 = np.dot(W, x_0) - alpha*y

            temp2 = grad_full(A_list, b_list, x_1)
                    
            y = np.dot(W, y) + temp2 - temp1

        else:
        
            x_1 = x_0 - alpha*y

            temp2 = grad_full(A_list, b_list, x_1)
                    
            y = y + temp2 - temp1
            
        x_0 = x_1

        temp1 = temp2
        
        norms[i] = np.linalg.norm(np.mean(grad_full(A_list, b_list, np.tile(np.mean(x_1, 0).T, (np.shape(W)[0], 1))), 0))  
            
    return norms

def DIGing_R_track(x_0, W, iterations, A_list, b_list, A, b, alpha, grad, grad_full, p, seed, x_opt):
        
    np.random.seed(seed)

    temp1 = grad_full(A_list, b_list, x_0)

    y = temp1

    opt_error = np.zeros(iterations)
    con_x_error = np.zeros(iterations)
    
    for i in tqdm(range(iterations)):        
        
        if np.random.rand() <= p:
        
            x_1 = np.dot(W, x_0) - alpha*y

            temp2 = grad_full(A_list, b_list, x_1)
                    
            y = np.dot(W, y) + temp2 - temp1

        else:
        
            x_1 = x_0 - alpha*y

            temp2 = grad_full(A_list, b_list, x_1)
                    
            y = y + temp2 - temp1
            
        x_0 = x_1

        temp1 = temp2

        opt_error[i] = np.linalg.norm(np.mean(x_1, 0)[np.newaxis].T - x_opt)

        con_x_error[i] = np.linalg.norm(np.mean(x_1, 0) - x_1)
            
    return opt_error, con_x_error