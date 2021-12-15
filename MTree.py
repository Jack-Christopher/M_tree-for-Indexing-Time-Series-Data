""" M-tree para hacer busqueda de los vecinos mas cercanos (mas similares) 
a un objeto dado. Se permite la insercion de elemento repetidos """

import abc
from heapq import heappush, heappop
import collections
from itertools import combinations, islice

def promotion_for_non_root(entries, current_routing_entry, d):
    if current_routing_entry is None or any(e.distance_to_parent is None for e in entries):
        return promotion_for_root(entries, current_routing_entry, d)
    
    new_entry = max(entries, key=lambda e: e.distance_to_parent)
    return (current_routing_entry.obj, new_entry.obj)

def promotion_for_root(entries, unused_current_routing_entry, d):
    objs = map(lambda e: e.obj, entries)
    return max(combinations(objs, 2), key=lambda two_objs: d(*two_objs))

def partition_process(entries, routing_object1, routing_object2, d):
    partition = (set(), set())
    for entry in entries:
        partition[d(entry.obj, routing_object1) > d(entry.obj, routing_object2)].add(entry)
    if not partition[0] or not partition[1]:
        partition = (set(islice(entries, len(entries)//2)), set(islice(entries, len(entries)//2, len(entries))))
    return partition


class MTree(object):
    def __init__(self, d, max_node_size=4, promote=promotion_for_non_root, partition=partition_process):
        if not callable(d): # si d no es una funcion
            raise TypeError('El parametro d debe ser una funcion')
        if max_node_size < 2:
            raise ValueError('max_node_size debe ser mayor o igual a 2')
        self.d = d
        self.max_node_size = max_node_size
        self.promote = promote
        self.partition = partition
        self.size = 0
        self.root = LeafNode(self)

    def __len__(self):
        return self.size

    def add(self, obj):
        # Agrega un valor al M-tree
        self.root.add(obj)
        self.size += 1

    def add_all(self, iterable):
        # Agrega todos los elementos al M-tree
        for obj in iterable:
            self.add(obj)

    def search(self, query_obj, k=1):
        k = min(k, len(self))
        if k == 0: return []
        # cola de prioridad de subarboles aun no explorados, ordenados por dmin
        pr = []
        heappush(pr, PrEntry(self.root, 0, 0))
        nn = NN(k)

        while pr:
            prEntry = heappop(pr)
            # el mejor candidato esta muy lejos, se detiene al no poder obtener una mejor respuesta
            if(prEntry.dmin > nn.search_radius()):
                break 
            prEntry.tree.search(query_obj, pr, nn, prEntry.d_query)
        return nn.result_list()
    
NNEntry = collections.namedtuple('NNEntry', 'obj dmax')
class NN(object):
    def __init__(self, size):
        self.elems = [NNEntry(None, float("inf"))] * size
        self.dmax = float("inf")

    def __len__(self):
        return len(self.elems)

    # El radio de búsqueda del  k-nn. también conocido como dmax es dinámico.
    def search_radius(self):
        return self.dmax

    def update(self, obj, dmax):
        if obj == None:
            #internal node
            self.dmax = min(self.dmax, dmax)
            return
        self.elems.append(NNEntry(obj, dmax))
        for i in range(len(self)-1, 0, -1):
            if self.elems[i].dmax < self.elems[i-1].dmax:
                self.elems[i-1], self.elems[i] = self.elems[i], self.elems[i-1]
            else:
                break
        self.elems.pop()

    def result_list(self):
        result = map(lambda entry: entry.obj, self.elems)
        return result

    def __repr__(self):
        return "NN(%r)" % self.elems
            

class PrEntry(object):
    def __init__(self, tree, dmin, d_query):
        self.tree = tree
        self.dmin = dmin
        self.d_query = d_query  # d_query: distancia al objetobuscado

    def __lt__(self, other):
        return self.dmin < other.dmin

    def __repr__(self):
        return "PrEntry(tree:%r, dmin:%r)" % (self.tree, self.dmin)

    
class Entry(object):
    def __init__(self, obj, distance_to_parent=None, radius=None, subtree=None):
        self.obj = obj
        self.distance_to_parent = distance_to_parent
        self.radius = radius
        self.subtree = subtree

    def __repr__(self):
        return "Entry(obj: %r, dist: %r, radius: %r, subtree: %r)" % (
            self.obj, self.distance_to_parent, self.radius,
            self.subtree.repr_class() if self.subtree else self.subtree)


class AbstractNode(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, mtree, parent_node=None, parent_entry=None, entries=None):
        self.mtree = mtree
        self.parent_node = parent_node
        self.parent_entry = parent_entry
        self.entries = set(entries) if entries else set()

    def __repr__(self):
        entries_str = '%s' % list(islice(self.entries, 2))
        if len(self.entries) > 2:
            entries_str = entries_str[:-1] + ', ...]'
            
        return "%s(parent_node: %s, parent_entry: %s, entries:%s)" % (
            self.__class__.__name__, self.parent_node.repr_class() if self.parent_node else self.parent_node,
            self.parent_entry, entries_str )

    def repr_class(self):
        return self.__class__.__name__ + "()"

    def __len__(self): 
        return len(self.entries)

    @property
    def d(self):
        return self.mtree.d

    def is_full(self):
        return len(self) == self.mtree.max_node_size

    def is_empty(self):
        return len(self) == 0

    def is_root(self):
        return self is self.mtree.root

    def remove_entry(self, entry):
        self.entries.remove(entry)

    def add_entry(self, entry):
        if self.is_full():
            raise ValueError('Se intenta agregar %s a un nodo lleno' % str(entry))
        self.entries.add(entry)

    def set_entries_and_parent_entry(self, new_entries, new_parent_entry):
        self.entries = new_entries
        self.parent_entry = new_parent_entry
        self.parent_entry.radius = self.covering_radius_for(self.parent_entry.obj)
        self._update_entries_distance_to_parent()

    def _update_entries_distance_to_parent(self):
        if self.parent_entry:
            for entry in self.entries:
                entry.distance_to_parent = self.d(entry.obj, self.parent_entry.obj)

    @abc.abstractmethod
    def add(self, obj): # Agrega 'obj' a este subarbol
        pass

    @abc.abstractmethod         
    def covering_radius_for(self, obj): # Calcula el 'radius' necesario para que 'obj' cubra las entradas de este nodo
        pass

    @abc.abstractmethod
    def search(self, query_obj, pr, nn, d_parent_query):
        pass

class LeafNode(AbstractNode):   # Nodo hoja del M-tree
    def __init__(self, mtree, parent_node=None, parent_entry=None, entries=None):
        AbstractNode.__init__(self, mtree, parent_node, parent_entry, entries)
    def add(self, obj):
        distance_to_parent = self.d(obj, self.parent_entry.obj) if self.parent_entry else None
        new_entry = Entry(obj, distance_to_parent)
        if not self.is_full():
            self.entries.add(new_entry)
        else:
            split(self, new_entry, self.d)
        assert self.is_root() or self.parent_node        

    def covering_radius_for(self, obj):
        if not self.entries:
            return 0
        else:
            return max(map(lambda e: self.d(obj, e.obj), self.entries))

    # Determina sin ningún cálculo d si podría haber objetos en el subárbol que pertenezcan al resultado
    def could_contain_results(self, query_obj, search_radius, distance_to_parent, d_parent_query):
        if self.is_root():
            return True
        return abs(d_parent_query - distance_to_parent) <= search_radius
        
    def search(self, query_obj, pr, nn, d_parent_query):
        for entry in self.entries:
            if self.could_contain_results(query_obj, nn.search_radius(), entry.distance_to_parent, d_parent_query):
                distance_entry_to_q = self.d(entry.obj, query_obj)
                if distance_entry_to_q <= nn.search_radius():
                    nn.update(entry.obj, distance_entry_to_q)
    
class InternalNode(AbstractNode):   # Nodo interno del M-tree
    def __init__(self, mtree, parent_node=None, parent_entry=None, entries=None):
        AbstractNode.__init__(self, mtree, parent_node,  parent_entry, entries)

    def add(self, obj): # Alamacena el 'obj' en un diccionario para evitar calcularlo de nuevo
        dist_to_obj = {}
        for entry in self.entries:
            dist_to_obj[entry] = self.d(obj, entry.obj)

        def find_best_entry_requiring_no_covering_radius_increase():
            valid_entries = [e for e in self.entries if dist_to_obj[e] <= e.radius]
            return min(valid_entries, key=dist_to_obj.get) if valid_entries else None
                
        def find_best_entry_minimizing_radius_increase():
            entry = min(self.entries, key=lambda e: dist_to_obj[e] - e.radius)
            # Agranda el radio de modo que 'obj' logre estar en el radio de cobertura de e
            entry.radius = dist_to_obj[entry]
            return entry

        entry = find_best_entry_requiring_no_covering_radius_increase() or \
            find_best_entry_minimizing_radius_increase()
        entry.subtree.add(obj)
        assert self.is_root() or self.parent_node

    # Calcula el radio minimo de 'obj' para que cubra los radios de todos los objetos de enrutamiento de este nodo
    def covering_radius_for(self, obj):
        if not self.entries:
            return 0
        else:
            return max(map(lambda e: self.d(obj, e.obj) + e.radius, self.entries))

    def set_entries_and_parent_entry(self, new_entries, new_parent_entry):
        AbstractNode.set_entries_and_parent_entry(self, new_entries, new_parent_entry)
        for entry in self.entries:
            entry.subtree.parent_node = self

    # Determina sin ningun calculo si podria haber objetos en el subarbol que pertenezcan al resultado
    def could_contain_results(self, query_obj, search_radius, entry, d_parent_query):
        if self.is_root():
            return True        
        parent_obj = self.parent_entry.obj
        return abs(d_parent_query - entry.distance_to_parent) <= search_radius + entry.radius
            
    def search(self, query_obj, pr, nn, d_parent_query):
        for entry in self.entries:
            if self.could_contain_results(query_obj, nn.search_radius(), entry, d_parent_query):
                d_entry_query = self.d(entry.obj, query_obj)
                entry_dmin = max(d_entry_query - entry.radius, 0)
                if entry_dmin <= nn.search_radius():
                    heappush(pr, PrEntry(entry.subtree, entry_dmin, d_entry_query))
                    entry_dmax = d_entry_query + entry.radius
                    if entry_dmax < nn.search_radius():
                        nn.update(None, entry_dmax)
                        

def split(existing_node, entry, d):
    assert existing_node.is_full()
    mtree = existing_node.mtree
    new_node = type(existing_node)(existing_node.mtree)
    all_entries = existing_node.entries | set((entry,))

    routing_object1, routing_object2 = mtree.promote(all_entries, existing_node.parent_entry, d)
    entries1, entries2 = mtree.partition(all_entries, routing_object1, routing_object2, d)
    assert entries1 and entries2, "Error during split operation. All the entries have been assigned to one routing_objects and none to the other! Should never happen since at least the routing objects are assigned to their corresponding set of entries"
    
    # Guarda la entrada anterior del nodo existente para eliminarse del nodo padre despues
    old_existing_node_parent_entry = existing_node.parent_entry

    existing_node_entry = Entry(routing_object1, None, None, existing_node)    
    existing_node.set_entries_and_parent_entry(entries1, existing_node_entry)

    new_node_entry = Entry(routing_object2, None, None, new_node)
    new_node.set_entries_and_parent_entry(entries2, new_node_entry)
                                          
    if existing_node.is_root():
        new_root_node = InternalNode(existing_node.mtree)

        existing_node.parent_node = new_root_node
        new_root_node.add_entry(existing_node_entry)
        
        new_node.parent_node = new_root_node
        new_root_node.add_entry(new_node_entry)
        
        mtree.root = new_root_node
    else:
        parent_node = existing_node.parent_node

        if not parent_node.is_root():
            # nodo padre tiene un padre en si mismo, entonces las dos entradas que agregamos
            # en el padre debe tener la distance_to_parent  establecida apropiadamente
            existing_node_entry.distance_to_parent = d(existing_node_entry.obj, parent_node.parent_entry.obj)
            new_node_entry.distance_to_parent = d(new_node_entry.obj, parent_node.parent_entry.obj)

        parent_node.remove_entry(old_existing_node_parent_entry)
        parent_node.add_entry(existing_node_entry)
        
        if parent_node.is_full():
            split(parent_node, new_node_entry, d)
        else:
            parent_node.add_entry(new_node_entry)
            new_node.parent_node = parent_node
    assert existing_node.is_root() or existing_node.parent_node
    assert new_node.is_root() or new_node.parent_node