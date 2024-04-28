import numpy as np
from tqdm import tqdm

def DIGing_2R(x_0, W, iterations, A_list, b_list, A, b, alpha, grad, grad_full, p, seed):

    np.random.seed(seed)
    
    x_0_1 = np.dot(W, x_0)
    temp1 =  grad_full(A_list, b_list, x_0_1)

    y = temp1

    norms = np.zeros(iterations)
    
    for i in tqdm(range(iterations)):

        if np.random.rand() <= p:
        
            x_1 = x_0_1 - alpha*y
            
            x_1_1 = np.dot(W, x_1)

            temp2 = grad_full(A_list, b_list, x_1_1)
            
            y = np.dot(W, y) + temp2 - temp1

        else:

            x_1_1 = x_0_1 - alpha*y
            
            temp2 = grad_full(A_list, b_list, x_1_1)
            
            y = y + temp2 - temp1
        
        x_0_1 = x_1_1

        temp1 = temp2
        
        norms[i] = np.linalg.norm(np.mean(grad_full(A_list, b_list, np.tile(np.mean(x_1_1, 0).T, (np.shape(W)[0], 1))), 0))
                        
    return norms

def DIGing_2R_track(x_0, W, iterations, A_list, b_list, A, b, alpha, grad, grad_full, p, seed, x_opt):
    
    np.random.seed(seed)
    x_0_1 = np.dot(W, x_0)
    temp1 =  grad_full(A_list, b_list, x_0_1)

    y = temp1
 
    opt_error = np.zeros(iterations)
    con_x_error = np.zeros(iterations)
    
    for i in tqdm(range(iterations)):
        
        
        if np.random.rand() <= p:
        
            x_1 = x_0_1 - alpha*y
            
            x_1_1 = np.dot(W, x_1)

            temp2 = grad_full(A_list, b_list, x_1_1)
            
            y = np.dot(W, y) + temp2 - temp1

        else:

            x_1_1 = x_0_1 - alpha*y
            
            temp2 = grad_full(A_list, b_list, x_1_1)
            
            y = y + temp2 - temp1
        
        x_0_1 = x_1_1

        temp1 = temp2

        opt_error[i] = np.linalg.norm(np.mean(x_0_1, 0)[np.newaxis].T - x_opt)

        con_x_error[i] = np.linalg.norm(np.mean(x_0_1, 0) - x_0_1)
            
    # return norms, opt_error, con_x_error, con_y_error, f
    return opt_error, con_x_error