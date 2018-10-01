from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tempfile
import pickle


from os import remove
from copy import deepcopy
from operator import itemgetter
try:
    from numpy import array
    from scipy import sparse
    from sklearn.datasets import load_svmlight_file
    from sklearn import svm
except ImportError:
    pass

from nltk.parse.transitionparser import Transition, Configuration, TransitionParser
from nltk.parse import ParserI, DependencyGraph, DependencyEvaluator

class MyTransitionParser(ParserI):

    """
    Class for transition based parser. Implement 2 algorithms which are "arc-standard" and "arc-eager"
    """
    ARC_STANDARD = 'arc-standard'
    ARC_EAGER = 'arc-eager'

    def __init__(self, algorithm,features_list=None):
        """
        :param algorithm: the algorithm option of this parser. Currently support `arc-standard` and `arc-eager` algorithm
        :type algorithm: str
        """
        if not(algorithm in [self.ARC_STANDARD, self.ARC_EAGER]):
            raise ValueError(" Currently we only support %s and %s " %
                                        (self.ARC_STANDARD, self.ARC_EAGER))
        self._algorithm = algorithm

        self._dictionary = {}
        self._transition = {}
        self._match_transition = {}
        self._features_list=features_list

    def _get_dep_relation(self, idx_parent, idx_child, depgraph):
        p_node = depgraph.nodes[idx_parent]
        c_node = depgraph.nodes[idx_child]

        if c_node['word'] is None:
            return None  # Root word

        if c_node['head'] == p_node['address']:
            return c_node['rel']
        else:
            return None

    def _convert_to_binary_features(self, features):
        """
        :param features: list of feature string which is needed to convert to binary features
        :type features: list(str)
        :return : string of binary features in libsvm format  which is 'featureID:value' pairs
        """
        unsorted_result = []
        for feature in features:
            self._dictionary.setdefault(feature, len(self._dictionary))
            unsorted_result.append(self._dictionary[feature])

        # Default value of each feature is 1.0
        return ' '.join(str(featureID) + ':1.0' for featureID in sorted(unsorted_result))

    def _is_projective(self, depgraph):
        arc_list = []
        for key in depgraph.nodes:
            node = depgraph.nodes[key]

            if 'head' in node:
                childIdx = node['address']
                parentIdx = node['head']
                if parentIdx is not None:
                    arc_list.append((parentIdx, childIdx))

        for (parentIdx, childIdx) in arc_list:
            # Ensure that childIdx < parentIdx
            if childIdx > parentIdx:
                temp = childIdx
                childIdx = parentIdx
                parentIdx = temp
            for k in range(childIdx + 1, parentIdx):
                for m in range(len(depgraph.nodes)):
                    if (m < childIdx) or (m > parentIdx):
                        if (k, m) in arc_list:
                            return False
                        if (m, k) in arc_list:
                            return False
        return True

    def _write_to_file(self, key, binary_features, input_file):
        """
        write the binary features to input file and update the transition dictionary
        """
        self._transition.setdefault(key, len(self._transition) + 1)
        self._match_transition[self._transition[key]] = key

        input_str = str(self._transition[key]) + ' ' + binary_features + '\n'
        input_file.write(input_str.encode('utf-8'))

    def _create_training_examples_arc_std(self, depgraphs, input_file):
        """
        Create the training example in the libsvm format and write it to the input_file.
        Reference : Page 32, Chapter 3. Dependency Parsing by Sandra Kubler, Ryan McDonal and Joakim Nivre (2009)
        """
        operation = Transition(self.ARC_STANDARD)
        count_proj = 0
        training_seq = []

        for depgraph in depgraphs:
            if not self._is_projective(depgraph):
                continue

            count_proj += 1
            conf = MyConfiguration(depgraph)
            while len(conf.buffer) > 0:
                b0 = conf.buffer[0]
                features = conf.extract_features(features_list=self._features_list)
                binary_features = self._convert_to_binary_features(features)

                if len(conf.stack) > 0:
                    s0 = conf.stack[len(conf.stack) - 1]
                    # Left-arc operation
                    rel = self._get_dep_relation(b0, s0, depgraph)
                    if rel is not None:
                        key = Transition.LEFT_ARC + ':' + rel
                        self._write_to_file(key, binary_features, input_file)
                        operation.left_arc(conf, rel)
                        training_seq.append(key)
                        continue

                    # Right-arc operation
                    rel = self._get_dep_relation(s0, b0, depgraph)
                    if rel is not None:
                        precondition = True
                        # Get the max-index of buffer
                        maxID = conf._max_address

                        for w in range(maxID + 1):
                            if w != b0:
                                relw = self._get_dep_relation(b0, w, depgraph)
                                if relw is not None:
                                    if (b0, relw, w) not in conf.arcs:
                                        precondition = False

                        if precondition:
                            key = Transition.RIGHT_ARC + ':' + rel
                            self._write_to_file(
                                key,
                                binary_features,
                                input_file)
                            operation.right_arc(conf, rel)
                            training_seq.append(key)
                            continue

                # Shift operation as the default
                key = Transition.SHIFT
                self._write_to_file(key, binary_features, input_file)
                operation.shift(conf)
                training_seq.append(key)

        print(" Number of training examples : " + str(len(depgraphs)))
        print(" Number of valid (projective) examples : " + str(count_proj))
        return training_seq

    def _create_training_examples_arc_eager(self, depgraphs, input_file):
        """
        Create the training example in the libsvm format and write it to the input_file.
        Reference : 'A Dynamic Oracle for Arc-Eager Dependency Parsing' by Joav Goldberg and Joakim Nivre
        """
        operation = Transition(self.ARC_EAGER)
        countProj = 0
        training_seq = []

        for depgraph in depgraphs:
            if not self._is_projective(depgraph):
                continue

            countProj += 1
            conf = MyConfiguration(depgraph)
            while len(conf.buffer) > 0:
                b0 = conf.buffer[0]
                features = conf.extract_features(features_list=self._features_list)
                binary_features = self._convert_to_binary_features(features)

                if len(conf.stack) > 0:
                    s0 = conf.stack[len(conf.stack) - 1]
                    # Left-arc operation
                    rel = self._get_dep_relation(b0, s0, depgraph)
                    if rel is not None:
                        key = Transition.LEFT_ARC + ':' + rel
                        self._write_to_file(key, binary_features, input_file)
                        operation.left_arc(conf, rel)
                        training_seq.append(key)
                        continue

                    # Right-arc operation
                    rel = self._get_dep_relation(s0, b0, depgraph)
                    if rel is not None:
                        key = Transition.RIGHT_ARC + ':' + rel
                        self._write_to_file(key, binary_features, input_file)
                        operation.right_arc(conf, rel)
                        training_seq.append(key)
                        continue

                    # reduce operation
                    flag = False
                    for k in range(s0):
                        if self._get_dep_relation(k, b0, depgraph) is not None:
                            flag = True
                        if self._get_dep_relation(b0, k, depgraph) is not None:
                            flag = True
                    if flag:
                        key = Transition.REDUCE
                        self._write_to_file(key, binary_features, input_file)
                        operation.reduce(conf)
                        training_seq.append(key)
                        continue

                # Shift operation as the default
                key = Transition.SHIFT
                self._write_to_file(key, binary_features, input_file)
                operation.shift(conf)
                training_seq.append(key)

        print(" Number of training examples : " + str(len(depgraphs)))
        print(" Number of valid (projective) examples : " + str(countProj))
        return training_seq


    def train(self, depgraphs, modelfile, verbose=True,model=None):
        """
        :param depgraphs : list of DependencyGraph as the training data
        :type depgraphs : DependencyGraph
        :param modelfile : file name to save the trained model
        :type modelfile : str
        """

        try:
            input_file = tempfile.NamedTemporaryFile(
                prefix='transition_parse.train',
                dir=tempfile.gettempdir(),
                delete=False)

            if self._algorithm == self.ARC_STANDARD:
                self._create_training_examples_arc_std(depgraphs, input_file)
            else:
                self._create_training_examples_arc_eager(depgraphs, input_file)

            input_file.close()
            # Using the temporary file to train the libsvm classifier
            x_train, y_train = load_svmlight_file(input_file.name)
            # The parameter is set according to the paper:
            # Algorithms for Deterministic Incremental Dependency Parsing by Joakim Nivre
            # Todo : because of probability = True => very slow due to
            # cross-validation. Need to improve the speed here
            if model is None:
                model = svm.SVC(
                    kernel='poly',
                    degree=2,
                    coef0=0,
                    gamma=0.2,
                    C=0.5,
                    verbose=verbose,
                    probability=True)

            model.fit(x_train, y_train)
            # Save the model to file name (as pickle)
            pickle.dump(model, open(modelfile, 'wb'))
        finally:
            remove(input_file.name)



    def parse(self, depgraphs, modelFile):
        """
        :param depgraphs: the list of test sentence, each sentence is represented as a dependency graph where the 'head' information is dummy
        :type depgraphs: list(DependencyGraph)
        :param modelfile: the model file
        :type modelfile: str
        :return: list (DependencyGraph) with the 'head' and 'rel' information
        """
        result = []
        # First load the model
        model = pickle.load(open(modelFile, 'rb'))
        operation = Transition(self._algorithm)

        for depgraph in depgraphs:
            conf = MyConfiguration(depgraph)
            while len(conf.buffer) > 0:
                features = conf.extract_features(features_list=self._features_list)
                col = []
                row = []
                data = []
                for feature in features:
                    if feature in self._dictionary:
                        col.append(self._dictionary[feature])
                        row.append(0)
                        data.append(1.0)
                np_col = array(sorted(col))  # NB : index must be sorted
                np_row = array(row)
                np_data = array(data)

                x_test = sparse.csr_matrix((np_data, (np_row, np_col)), shape=(1, len(self._dictionary)))

                # It's best to use decision function as follow BUT it's not supported yet for sparse SVM
                # Using decision funcion to build the votes array
                #dec_func = model.decision_function(x_test)[0]
                #votes = {}
                #k = 0
                # for i in range(len(model.classes_)):
                #    for j in range(i+1, len(model.classes_)):
                #        #if  dec_func[k] > 0:
                #            votes.setdefault(i,0)
                #            votes[i] +=1
                #        else:
                #           votes.setdefault(j,0)
                #           votes[j] +=1
                #        k +=1
                # Sort votes according to the values
                #sorted_votes = sorted(votes.items(), key=itemgetter(1), reverse=True)

                # We will use predict_proba instead of decision_function
                prob_dict = {}
                pred_prob = model.predict_proba(x_test)[0]
                for i in range(len(pred_prob)):
                    prob_dict[i] = pred_prob[i]
                sorted_Prob = sorted(
                    prob_dict.items(),
                    key=itemgetter(1),
                    reverse=True)

                # Note that SHIFT is always a valid operation
                for (y_pred_idx, confidence) in sorted_Prob:
                    #y_pred = model.predict(x_test)[0]
                    # From the prediction match to the operation
                    y_pred = model.classes_[y_pred_idx]

                    if y_pred in self._match_transition:
                        strTransition = self._match_transition[y_pred]
                        baseTransition = strTransition.split(":")[0]

                        if baseTransition == Transition.LEFT_ARC:
                            if operation.left_arc(conf, strTransition.split(":")[1]) != -1:
                                break
                        elif baseTransition == Transition.RIGHT_ARC:
                            if operation.right_arc(conf, strTransition.split(":")[1]) != -1:
                                break
                        elif baseTransition == Transition.REDUCE:
                            if operation.reduce(conf) != -1:
                                break
                        elif baseTransition == Transition.SHIFT:
                            if operation.shift(conf) != -1:
                                break
                    else:
                        raise ValueError("The predicted transition is not recognized, expected errors")

            # Finish with operations build the dependency graph from Conf.arcs

            new_depgraph = deepcopy(depgraph)
            for key in new_depgraph.nodes:
                node = new_depgraph.nodes[key]
                node['rel'] = ''
                # With the default, all the token depend on the Root
                node['head'] = 0
            for (head, rel, child) in conf.arcs:
                c_node = new_depgraph.nodes[child]
                c_node['head'] = head
                c_node['rel'] = rel
            result.append(new_depgraph)

        return result




################################################################################################################################

class MyConfiguration(object):
    """
    Class for holding configuration which is the partial analysis of the input sentence.
    The transition based parser aims at finding set of operators that transfer the initial
    configuration to the terminal configuration.

    The configuration includes:
        - Stack: for storing partially proceeded words
        - Buffer: for storing remaining input words
        - Set of arcs: for storing partially built dependency tree

    This class also provides a method to represent a configuration as list of features.
    """

    def __init__(self, dep_graph):
        """
        :param dep_graph: the representation of an input in the form of dependency graph.
        :type dep_graph: DependencyGraph where the dependencies are not specified.
        """
        # dep_graph.nodes contain list of token for a sentence
        self.stack = [0]  # The root element
        self.buffer = list(range(1, len(dep_graph.nodes)))  # The rest is in the buffer
        self.arcs = []  # empty set of arc
        self._tokens = dep_graph.nodes
        self._max_address = len(self.buffer)

    def __str__(self):
        return 'Stack : ' + \
            str(self.stack) + '  Buffer : ' + str(self.buffer) + '   Arcs : ' + str(self.arcs)

    def _check_informative(self, feat, flag=False):
        """
        Check whether a feature is informative
        The flag control whether "_" is informative or not
        """
        if feat is None:
            return False
        if feat == '':
            return False
        if flag is False:
            if feat == '_':
                return False
        return True

    def extract_features(self,features_list=None):
        """
        Extract the set of features for the current configuration. Implement standard features as describe in
        Table 3.2 (page 31) in Dependency Parsing book by Sandra Kubler, Ryan McDonal, Joakim Nivre.
        Please note that these features are very basic.
        :return: list(str)
        """
        result = []
        # Todo : can come up with more complicated features set for better
        # performance.
        if len(self.stack) > 0:
            # Stack 0
            stack_idx0 = self.stack[len(self.stack) - 1]
            token = self._tokens[stack_idx0]
            if self._check_informative(token['word'], True):
                result.append('STK_0_FORM_' + token['word'])
            if 'lemma' in token and self._check_informative(token['lemma']):
                result.append('STK_0_LEMMA_' + token['lemma'])
            if self._check_informative(token['tag']):
                result.append('STK_0_POS_' + token['tag'])
            if 'feats' in token and self._check_informative(token['feats']) and (features_list is None or 'feats' in features_list):
                feats = token['feats'].split("|")
                for feat in feats:
                    result.append('STK_0_FEATS_' + feat)
            # Stack 1
            if len(self.stack) > 1:
                stack_idx1 = self.stack[len(self.stack) - 2]
                token = self._tokens[stack_idx1]
                if self._check_informative(token['tag']):
                    result.append('STK_1_POS_' + token['tag'])

            # Left most, right most dependency of stack[0]
            left_most = 1000000
            right_most = -1
            dep_left_most = ''
            dep_right_most = ''
            for (wi, r, wj) in self.arcs:
                if wi == stack_idx0:
                    if (wj > wi) and (wj > right_most):
                        right_most = wj
                        dep_right_most = r
                    if (wj < wi) and (wj < left_most):
                        left_most = wj
                        dep_left_most = r
            if self._check_informative(dep_left_most):
                result.append('STK_0_LDEP_' + dep_left_most)
            if self._check_informative(dep_right_most):
                result.append('STK_0_RDEP_' + dep_right_most)

        # Check Buffered 0
        if len(self.buffer) > 0:
            # Buffer 0
            buffer_idx0 = self.buffer[0]
            token = self._tokens[buffer_idx0]
            if self._check_informative(token['word'], True):
                result.append('BUF_0_FORM_' + token['word'])
            if 'lemma' in token and self._check_informative(token['lemma']):
                result.append('BUF_0_LEMMA_' + token['lemma'])
            if self._check_informative(token['tag']):
                result.append('BUF_0_POS_' + token['tag'])
            

            if 'feats' in token and self._check_informative(token['feats']) and (features_list is None or 'feats' in features_list):
                feats = token['feats'].split("|")
                for feat in feats:
                    result.append('BUF_0_FEATS_' + feat)  #### CHANGED HERE  -----  IMPORTANT
    

            # Buffer 1
            if len(self.buffer) > 1:
                buffer_idx1 = self.buffer[1]
                token = self._tokens[buffer_idx1]
                if self._check_informative(token['word'], True):
                    result.append('BUF_1_FORM_' + token['word'])
                if self._check_informative(token['tag']):
                    result.append('BUF_1_POS_' + token['tag'])
            if len(self.buffer) > 2:
                buffer_idx2 = self.buffer[2]
                token = self._tokens[buffer_idx2]
                if self._check_informative(token['tag']):
                    result.append('BUF_2_POS_' + token['tag'])
            if len(self.buffer) > 3:
                buffer_idx3 = self.buffer[3]
                token = self._tokens[buffer_idx3]
                if self._check_informative(token['tag']):
                    result.append('BUF_3_POS_' + token['tag'])
                    # Left most, right most dependency of stack[0]
            left_most = 1000000
            right_most = -1
            dep_left_most = ''
            dep_right_most = ''
            for (wi, r, wj) in self.arcs:
                if wi == buffer_idx0:
                    if (wj > wi) and (wj > right_most):
                        right_most = wj
                        dep_right_most = r
                    if (wj < wi) and (wj < left_most):
                        left_most = wj
                        dep_left_most = r
            if self._check_informative(dep_left_most):
                result.append('BUF_0_LDEP_' + dep_left_most)
            if self._check_informative(dep_right_most):
                result.append('BUF_0_RDEP_' + dep_right_most)

        return result