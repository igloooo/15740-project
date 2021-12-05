class Graph_Node:
    def __init__(self, *p):
        # if id = -1, it's the terminating node
        if len(p) == 1:
            self.id = p[0]
            self.size = -1
            self.branches = [None, None]
        elif len(p) == 2:
            self.id = p[0]
            self.size = p[1]
            self.branches = [None, None]
        elif len(p) == 3:
            self.id = p[0]
            self.size = p[1]
            self.branches = [p[2], None]
        elif len(p) == 4:
            self.id = p[0]
            self.size = p[1]
            self.branches = [p[2], p[3]]
        else:
            raise KeyError


class Order_Node:
    def __init__(self, *p):
        if len(p) == 1:
            self.id = p[0]
            self.prev = None
            self.next = None
        else:
            self.id = p[0]
            self.prev = p[1]
            self.next = p[2]


def traverse_ids(order_node, iters):
    """
    :param order_node: start traversing from this node
    :param iters: stop after iters
    :return: string containing the in of the order nodes
    """
    cat_str = ''
    for i in range(iters):
        cat_str = cat_str + '->' + str(order_node.id)
        order_node = order_node.next
    return cat_str

