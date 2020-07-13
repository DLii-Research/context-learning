import numpy as np

xor_data = [  
            [ np.array([[-1, -1]]),  0],
            [ np.array([[-1, 1]]), 1 ],
            [ np.array([[1, -1]]), 1 ],
            [ np.array([[1, 1]]), 0 ]
       ]


xnor_data = [  
            [ np.array([[-1, -1]]),  1],
            [ np.array([[-1, 1]]), 0 ],
            [ np.array([[1, -1]]), 0 ],
            [ np.array([[1, 1]]), 1 ]
       ]


and_data = [
            [ np.array([[-1, -1]]),  0],
            [ np.array([[-1, 1]]), 0 ],
            [ np.array([[1, -1]]), 0 ],
            [ np.array([[1, 1]]), 1 ]
        
]

or_data = [
            [ np.array([[-1, -1]]),  0],
            [ np.array([[-1, 1]]), 1 ],
            [ np.array([[1, -1]]), 1 ],
            [ np.array([[1, 1]]), 1 ]
        
]

nor_data = [
            [ np.array([[-1, -1]]),  1],
            [ np.array([[-1, 1]]), 0 ],
            [ np.array([[1, -1]]), 0 ],
            [ np.array([[1, 1]]), 0 ]
        
]

nand_data = [
            [ np.array([[-1, -1]]),  1],
            [ np.array([[-1, 1]]), 1 ],
            [ np.array([[1, -1]]), 1 ],
            [ np.array([[1, 1]]), 0 ]
        
]

custom_gate_0_data = [
            [ np.array([[-1, -1]]),  1],
            [ np.array([[-1, 1]]), 0 ],
            [ np.array([[1, -1]]), 1 ],
            [ np.array([[1, 1]]), 0 ]
        
]

custom_gate_1_data = [
            [ np.array([[-1, -1]]),  0],
            [ np.array([[-1, 1]]), 1 ],
            [ np.array([[1, -1]]), 0 ],
            [ np.array([[1, 1]]), 1 ]
        
]


all_data = [xor_data, xnor_data, and_data, or_data, nand_data, nor_data, custom_gate_0_data, custom_gate_1_data]
