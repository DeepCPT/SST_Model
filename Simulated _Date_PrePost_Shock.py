import torch
import torch.nn as nn
from scipy.stats import norm
import pickle
from collections import Counter
import matplotlib.pyplot as plt


# Universial setting of the padding value
pad_idx = -1e2
torch.manual_seed(0)

# 1. Generate data from Reserved Utility function from Kim et al. (2010)
def Get_C(R,V):
    PDF=norm.pdf(R-V)
    CDF=norm.cdf(R-V)
    C=(1-CDF)*((V-R)+PDF/(1-CDF)) # c should be positive
    return C

class ProductGenerator:
    def __init__(self, N_cate, binary_feature, continous_feature, total,seed=None):
        """
        Initializes the ProductGenerator.

        Parameters:
            N_cate (int): Number of categories.
            M (int): Number of meaningful features per category.
            binary_feature (int): Number of binary features in each product.
            continous_feature (int): Number of continous features in each product.
            total: Total number of products to generate.
        """
        self.N_cate = N_cate
        self.M = binary_feature+continous_feature
        self.binary_feature = binary_feature
        self.continous_feature = continous_feature
        self.total = total
        if seed is not None:
            torch.manual_seed(seed)

    def generate_products(self):
        """
        Generates X products distributed over N categories with specified feature properties.

        Returns:
            torch.Tensor: A tensor of shape (X, N * M + 1) where each row represents a product.
        """
        products = torch.zeros((self.total, self.N_cate * self.M + 1))
        products_per_category = self.total // self.N_cate

        for category in range(self.N_cate):
            start_idx = category * self.M
            end_idx = start_idx + self.M

            for i in range(products_per_category):
                idx = category * products_per_category + i

                # Generate M1 binary features and M - M1 continuous features for each product
                binary_features = torch.randint(0, 2, (self.binary_feature,))
                continuous_features = torch.rand(self.continous_feature)

                # Combine binary and continuous features for the product
                product_features = torch.cat([binary_features.float(), continuous_features])

                # Place product features in the appropriate category section
                products[idx, start_idx:end_idx] = product_features

                # Add a random price feature in the range [0, 1], this is the normalized version
                products[idx, -1] = torch.rand(1)

        return products


class ProductCostGenerator:
    def __init__(self, num_product,num_cost_feature,seed=None):
        self.num_product = num_product
        self.num_cost_feature = num_cost_feature
        if seed is not None:
            torch.manual_seed(seed)

    def generate_costs(self):
        cost_feature = -2*torch.rand(self.num_product,self.num_cost_feature)
        return cost_feature


class ProductPostClickFeatureGenerator:
    def __init__(self, num_product,num_post_click_feature,seed=None):
        self.num_product = num_product
        self.num_post_click_feature = num_post_click_feature
        if seed is not None:
            torch.manual_seed(seed)

    def generate_postclick_features(self):
        post_click_feature = torch.rand(self.num_product,self.num_post_click_feature)
        return post_click_feature

class CategorySampler:
    def __init__(self, products,product_costs,product_postclick, N_cate, num_product_per_cate, seed=None):
        self.products = products
        self.product_costs = product_costs
        self.product_postclick = product_postclick
        self.N_cate = N_cate
        self.num_product_per_cate = num_product_per_cate
        if seed is not None:
            torch.manual_seed(seed)

    def sample_from_category(self, category_index, number_sample):

        # Find the indices in the product matrix corresponding to the specified category
        start_idx = category_index * self.num_product_per_cate
        end_idx = start_idx + self.num_product_per_cate
        category_products = self.products[int(start_idx):int(end_idx)]
        category_products_costs = self.product_costs[int(start_idx):int(end_idx)]
        category_products_postclick = self.product_postclick[int(start_idx):int(end_idx)]

        # Ensure there are enough products in the chosen category
        if category_products.shape[0] < number_sample:
            raise ValueError(f"Not enough products in category {category_index} to sample {number_sample} items.")

        # Randomly sample Y products from the filtered category products
        sampled_indices = torch.randperm(category_products.shape[0])[:number_sample]
        sampled_products = category_products[sampled_indices]
        sampled_products_costs = category_products_costs[sampled_indices]
        sampled_products_postclick= category_products_postclick[sampled_indices]

        return sampled_products,sampled_products_costs,sampled_products_postclick


class PreferenceGenerator:
    def __init__(self, num_prefs, num_feature,products, lower_bound=0.25, seed=None):
        self.num_prefs = num_prefs
        self.products= products
        self.num_feature = num_feature
        self.lower_bound = lower_bound
        if seed is not None:
            torch.manual_seed(seed)

    def generate_preferences(self):
        """
        Generates num_prefs preferences, each with N * M + 1 elements, and adjusts
        the last element to achieve the target ratio of positive dot products.

        Returns:
            torch.Tensor: A tensor of shape (num_prefs, N * M + 1).
        """
        preferences = torch.normal(0, 1, size=(self.num_prefs, self.num_feature))
        return preferences


class CostCoeffGenerator:
    def __init__(self, num_consumer,num_cost_feature, target_positive_ratio=0.5, seed=None):
        self.num_consumer=num_consumer
        self.num_cost_feature = num_cost_feature
        self.target_positive_ratio = target_positive_ratio
        if seed is not None:
            torch.manual_seed(seed)

    def generate_cost(self, product,preference,list_length):
        cost_coeff = torch.normal(0, 1, size=(self.num_consumer, self.num_cost_feature))
        return cost_coeff


class PreferCostCoeffGenerator:
    def __init__(self, num_consumer,num_feature,num_prefer,num_cost,num_postclick_feature, seed=None):
        self.num_consumer=num_consumer
        self.num_feature = num_feature
        self.num_prefer = num_prefer
        self.num_cost = num_cost
        self.num_postclick = num_postclick_feature
        if seed is not None:
            torch.manual_seed(seed)

    def generate_coeff(self):
        while True:
            cost_coeff = torch.normal(0, 1, size=(self.num_consumer*500, self.num_feature))
            postclick_coeff = torch.normal(0, 0.2, size=(self.num_consumer * 500, self.num_postclick))
            # Mean threshold
            mean_threshold_lower=-0.2
            mean_threshold_upper = 0.2
            # Variance threshold
            var_threshold = 0.5
            cost_threshold = 0.5
            postclick_threshold =0.2 #0.2
            # Compute variance of each row
            prefer_mean = torch.mean(cost_coeff[:, :self.num_prefer], dim=1)
            prefer_vars = torch.var(cost_coeff[:,:self.num_prefer], dim=1)
            cost_vars = torch.var(cost_coeff[:,self.num_prefer:(self.num_prefer+self.num_cost)], dim=1)
            cost_mean = torch.mean(cost_coeff[:, self.num_prefer:(self.num_prefer + self.num_cost)], dim=1)
            postclick_vars = torch.var(postclick_coeff, dim=1)
            vars=prefer_vars+cost_vars

            # Get indices of rows with variance less than the threshold
            valid_indices_preferencecost = torch.nonzero(
                (vars < var_threshold) & (cost_vars < cost_threshold) & (cost_mean >= 0.1)&(cost_mean <= 0.5)& (prefer_mean >= mean_threshold_lower)& (prefer_mean <= mean_threshold_upper)).squeeze()
            valid_indices_postclick = torch.nonzero(postclick_vars < postclick_threshold).squeeze()

            if len(valid_indices_preferencecost)>=self.num_consumer and len(valid_indices_postclick)>=self.num_consumer:
                break

        # Randomly select x rows from valid indices (if available)
        selected_indices = valid_indices_preferencecost[torch.randperm(len(valid_indices_preferencecost))[:self.num_consumer]]
        selected_pre_cost = cost_coeff[selected_indices]
        selected_indices = valid_indices_postclick[torch.randperm(len(valid_indices_postclick))[:self.num_consumer]]
        select_postclick =postclick_coeff[selected_indices]


        return selected_pre_cost,select_postclick


def category_exposure_matrix(N_cate, Num_feature_per_Cate, expo_feature_num, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    # Initialize matrix with all False
    matrix = torch.zeros((N_cate, Num_feature_per_Cate), dtype=torch.bool)

    for row in range(N_cate):
        true_indices = torch.randperm(Num_feature_per_Cate - 1)[:expo_feature_num]  # Exclude the last index (x2-1)
        matrix[row, true_indices] = True
    return matrix


class PreTrainedModuleRU(nn.Module):
    "Mapping function from V,C to reserved utilities Z."
    def __init__(self):
        super(PreTrainedModuleRU, self).__init__()
        self.fc1 = nn.Linear(2, 16)  # Input layer to hidden layer
        self.fc2 = nn.Linear(16, 10)   # Hidden layer
        self.fc3 = nn.Linear(10, 8)  # Hidden layer
        self.fc4 = nn.Linear(8, 1)     # Hidden layer to output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Activation function
        x = torch.relu(self.fc2(x))  # Activation function
        x = torch.relu(self.fc3(x))  # Activation function
        x = self.fc4(x)               # Output layer
        return x

class PreClickShock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PreClickShock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # Input layer to hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)   # Hidden layer
        self.fc3 = nn.Linear(hidden_dim, output_dim)  # Hidden layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Activation function
        x = torch.relu(self.fc2(x))  # Activation function
        x = self.fc3(x)              # Output layer
        return x

class ConsumerClick:
    def __init__(self, preferences, cost_coeff,post_click_coeff, N_cate, num_feature, num_cost_feature, num_postclick_feature,category_sampler, ModuleRU,
                 list_length,preference_shock_var, seed=None):
        self.preferences = preferences
        self.num_categories = N_cate
        self.num_features = num_feature
        self.num_cost_feature = num_cost_feature
        self.num_postclick_feature = num_postclick_feature
        self.product_sampler = category_sampler
        self.cost_coeff = cost_coeff
        self.post_click_coeff = post_click_coeff
        self.ModuleRU = ModuleRU
        self.list_length = list_length
        self.preference_shock_var = preference_shock_var
        if seed is not None:
            torch.manual_seed(seed)

    def preference_shock(self, pre_session,rnn):

        pre_session_length=pre_session.shape[1]
        feature_length = pre_session.shape[0]
        shock_length=feature_length-2

        h_0 = torch.zeros(1, shock_length)
        output, h_n = rnn(pre_session.t(), h_0)
        shock=h_n.squeeze()


        mean = shock.mean()
        std = shock.std()
        shock= (shock - mean) / std
        return shock*(self.preference_shock_var ** 0.5)


    def simulate_consecutive_sessions(self,num_records):
        # Data need to return
        consumer_id_list=[]
        original_preference_list=[]
        updated_preference_list = []
        cost_coeff_list=[]
        post_click_coeff_list=[]
        first_category_list=[]
        second_category_list = []
        category_preferences_list=[]
        X = []
        Y = []
        Post_Click=[]
        Click_Index=[]
        Expo = []
        Expo_Cost = []
        purchase = []
        tgt_len = []
        pre_utility = []
        full_utility = []
        shock_list=[]
        preference_shock_flag = []

        rnn = nn.RNN(input_size=self.num_features+1, hidden_size=self.num_features-1, num_layers=1, batch_first=True)
        num_none_shock_sessions=torch.full((self.cost_coeff.size(0),), 20)

        for index in range(num_records):
            # Select a random preference for the consumer
            consumer_id = torch.randint(0, self.preferences.shape[0], (1,)).item()
            consumer_pref = self.preferences[consumer_id]
            cost_coeff=self.cost_coeff[consumer_id]
            post_click_coeff=self.post_click_coeff[consumer_id]

            # generate the first session
            first_session = None
            second_session = None
            non_shock_session = 0
            first_count=0
            while first_session is None:
                first_session=self.simulate_session_with_given_preference(consumer_pref, cost_coeff,post_click_coeff,'first')
                first_count=first_count+1
                if first_session is not None or first_count>100:
                    if first_count>100:
                        print("Exhaust 100 sessions for interation: "+str(index))
                    break
            if first_session is not None:
                # update the preference
                shock = self.preference_shock(first_session["x_candidate"], rnn)
                shock = torch.cat((shock, torch.tensor([0.])))
                if num_none_shock_sessions[consumer_id]==0:
                    #update_preference=consumer_pref.clone().detach()
                    update_preference=consumer_pref+shock
                else:
                    update_preference=consumer_pref
                    num_none_shock_sessions[consumer_id]=num_none_shock_sessions[consumer_id]-1
                    non_shock_session=1
                # generate the second session
                second_count=0
                while second_session is None:
                    second_session=self.simulate_session_with_given_preference(update_preference, cost_coeff,post_click_coeff,'second')
                    second_count=second_count+1
                    if second_session is not None or second_count>100:
                        if second_count>100:
                            print("Exhaust 100 sessions for interation: " + str(index))
                        break
            if first_session is not None and second_session is not None:
                # Record the data
                consumer_id_list.append(consumer_id)
                original_preference_list.append(consumer_pref)
                updated_preference_list.append(update_preference)
                shock_list.append(shock)
                cost_coeff_list.append(cost_coeff)
                post_click_coeff_list.append(post_click_coeff)
                first_category_list.append(first_session["category"])
                second_category_list.append(second_session["category"])

                X.append(first_session["x_candidate"])
                Y.append(second_session["y_candidate"])
                Post_Click.append(second_session["post_click_candidate"])
                Expo.append(second_session["expo_candidate"])
                Expo_Cost.append(second_session["expo_cost_candidate"])
                Click_Index.append(second_session["click_index_candidate"])
                purchase.append(second_session["purchase_mask"])
                tgt_len.append(second_session["tgt_len"])
                pre_utility.append(second_session["pre_click_utility_candidate"])
                full_utility.append(second_session["clicked_full_utility_candidate"])

                preference_shock_flag.append(non_shock_session)

        return {
            'consumer_id': consumer_id_list,
            "original_preference_list": original_preference_list,
            "updated_preference_list": updated_preference_list,
            "shock_list": shock_list,
            "cost_coeff": cost_coeff_list,
            "post_click_coeff_list": post_click_coeff_list,
            "first_category_list": first_category_list,
            "second_category_list": second_category_list,
            "X": X,
            "Y": Y,
            "Post_Click":Post_Click,
            "Expo": Expo,
            "Expo_Cost": Expo_Cost,
            "Click_Index":Click_Index,
            "purchase": purchase,
            "tgt_len": tgt_len,
            "pre_utility": pre_utility,
            "full_utility": full_utility,
            "preference_shock_flag": preference_shock_flag,

            "tgt_interaction_feature": Y[0].shape[0],
            "len_cost_coeff": self.cost_coeff.shape[1],
            "preference_shock_var": self.preference_shock_var,
            "num_features": self.num_features,
            "num_postclick_feature": self.num_postclick_feature,
            "num_consumer": self.preferences.shape[0]
        }

    def simulate_session_with_given_preference(self, preference,cost,post_click_coeff,session_index):

        # Randomly select a category
        list_length=self.list_length
        num_feature_per_cate = (self.num_features - 1) / self.num_categories
        random_category = torch.randint(0, self.num_categories, (1,)).item()

        # the start and end index of the choosen category
        start_index = int(random_category * num_feature_per_cate)
        end_index = int(start_index + num_feature_per_cate)

        # Set the masks
        category_mask = torch.zeros(self.num_features, dtype=torch.bool)
        category_mask[start_index:end_index] = True
        category_mask[-1] = True  # price is always visiable

        # Pre_Click_Preference and category_preferences
        consumer_pref = preference
        pre_click_preferences = consumer_pref
        category_preferences = pre_click_preferences * category_mask
        cost_coeff = cost

        # Generate products list for the chosen category
        original_list, original_cost_feature, original_products_postclick = self.product_sampler.sample_from_category(random_category, list_length)
        products_list = original_list.clone()
        cost_feature = original_cost_feature.clone()
        post_click_feature = original_products_postclick.clone()
        pre_click_error = torch.normal(0, 1, size=(list_length,)) / 10
        post_click_error = torch.normal(0, 1, size=(list_length,)) / 10
        pre_click_error.zero_()
        post_click_error.zero_()

        # candidate data, used to recod the required interactions, initialized at the beginning of the session
        x_candidate = torch.empty((self.num_features+1, 0)) # +1 represent the behavior of user, 1 means click and 2 is purchase
        y_candidate = torch.zeros((self.num_features)).unsqueeze(1)  # start with outside option, (another option is just using unexposed features)
        post_click_candidate=torch.zeros((self.num_postclick_feature)).unsqueeze(1)

        expo_candidate = torch.empty((0, list_length, self.num_features))
        expo_cost_candidate = torch.empty((0, list_length, self.num_cost_feature))
        click_index_candidate = torch.empty((0, list_length))
        pre_click_utility_candidate = torch.tensor([])
        clicked_full_utility_candidate = torch.tensor([])

        # Making sequence decisions
        current_list_length = list_length
        clicked_full_utility = []
        clicked_full_utility.append(torch.tensor(0))  # Add the outside option
        for i in range(list_length):
            # Get the cost of the list
            cost = torch.exp(torch.mv(cost_feature,cost_coeff))  # the cost transformation ensure the cost is positive, this should be same with the model setup

            # Get the post click utility of the list base on the current list
            post_click_utility = torch.mv(post_click_feature, post_click_coeff)

            # Get the reserved_utility of items in the list
            pre_click_utility = torch.mv(products_list, pre_click_preferences)
            pre_click_utility = pre_click_utility + pre_click_error

            reserved_utility = self.ModuleRU(torch.stack((pre_click_utility, cost), dim=-1)).squeeze()

            # choose the maximal element in the reserved_utility list
            max_reserved_utility, max_reserved_utility_index = torch.max(reserved_utility, dim=0)
            if max_reserved_utility.item() > max(clicked_full_utility) and i < list_length:  # keep search

                if session_index=='second':
                    clicked_item_utility = pre_click_utility[max_reserved_utility_index] + post_click_utility[max_reserved_utility_index]
                else:
                    clicked_item_utility = pre_click_utility[max_reserved_utility_index] + post_click_error[max_reserved_utility_index]

                clicked_full_utility.append(clicked_item_utility)

                x_candidate = torch.cat((x_candidate, (
                    torch.cat((products_list[max_reserved_utility_index], torch.tensor([1])))).unsqueeze(1)), dim=1)
                y_candidate = torch.cat((y_candidate, (products_list[max_reserved_utility_index]).unsqueeze(1)), dim=1)
                post_click_candidate = torch.cat((post_click_candidate, (post_click_feature[max_reserved_utility_index]).unsqueeze(1)), dim=1)

                index_vector = torch.zeros(list_length, dtype=torch.bool)
                index_vector[max_reserved_utility_index] = True
                click_index_candidate = torch.cat((click_index_candidate, index_vector.unsqueeze(-2)), dim=0)

                pre_click_utility_candidate = torch.cat(
                    (pre_click_utility_candidate, torch.tensor([pre_click_utility[max_reserved_utility_index]])))
                clicked_full_utility_candidate = torch.cat(
                    (clicked_full_utility_candidate, torch.tensor([clicked_item_utility])))

                # update expo list
                expo_list = products_list
                expo_cost_list = cost_feature
                if expo_list.shape[0] < list_length:  # padding the exposure list to list_length
                    append_matrix = torch.full((list_length - expo_list.shape[0], self.num_features),
                                               pad_idx)
                    expo_list = torch.cat((expo_list, append_matrix), dim=0)

                    cost_append_matrix = torch.full((list_length - expo_cost_list.shape[0], self.num_cost_feature),
                                                    pad_idx)
                    expo_cost_list = torch.cat((expo_cost_list, cost_append_matrix), dim=0)

                expo_candidate = torch.cat((expo_candidate, expo_list.unsqueeze(0)), dim=0)
                expo_cost_candidate = torch.cat((expo_cost_candidate, expo_cost_list.unsqueeze(0)), dim=0)

                # update the products_list and current_list_length
                if max_reserved_utility_index == products_list.size(0) - 1:
                    products_list = products_list[:max_reserved_utility_index]
                    cost_feature = cost_feature[:max_reserved_utility_index]
                    post_click_feature=post_click_feature[:max_reserved_utility_index]
                    post_click_error = post_click_error[:max_reserved_utility_index]
                    pre_click_error = pre_click_error[:max_reserved_utility_index]
                else:
                    products_list = torch.cat((products_list[:max_reserved_utility_index],
                                               products_list[max_reserved_utility_index + 1:]), dim=0)
                    cost_feature = torch.cat((cost_feature[:max_reserved_utility_index],
                                              cost_feature[max_reserved_utility_index + 1:]), dim=0)
                    post_click_feature = torch.cat((post_click_feature[:max_reserved_utility_index],
                                              post_click_feature[max_reserved_utility_index + 1:]), dim=0)
                    post_click_error = torch.cat((post_click_error[:max_reserved_utility_index],
                                                  post_click_error[max_reserved_utility_index + 1:]), dim=0)
                    pre_click_error = torch.cat((pre_click_error[:max_reserved_utility_index],
                                                 pre_click_error[max_reserved_utility_index + 1:]), dim=0)

                current_list_length = current_list_length - 1
            else:  # make the decision of which item to purchase
                # Find the maximum value
                max_clicked_value = max(clicked_full_utility)
                # Find the index of the maximum value
                max_clicked_index = clicked_full_utility.index(max_clicked_value)
                # set up the purchase mask
                purchase_mask = torch.zeros(i + 1)
                purchase_mask[max_clicked_index] = 1

                if max_clicked_index>0:
                    # update x_candidate about the purchased item
                    purchase_item_features = x_candidate[:, max_clicked_index - 1].clone()
                    purchase_item_features[-1] = 2  # the last element 2 means purchase
                    # add the purchased item to x
                    x_candidate = torch.cat((x_candidate, purchase_item_features.unsqueeze(-1)), dim=1)


                # update expo list
                expo_list = products_list
                expo_cost_list = cost_feature
                if expo_list.shape[0] < list_length:  # padding the exposure list to list_length
                    append_matrix = torch.full((list_length - expo_list.shape[0], self.num_features),
                                               pad_idx)
                    expo_list = torch.cat((expo_list, append_matrix), dim=0)

                    cost_append_matrix = torch.full((list_length - expo_cost_list.shape[0], self.num_cost_feature),
                                                    pad_idx)
                    expo_cost_list = torch.cat((expo_cost_list, cost_append_matrix), dim=0)

                expo_candidate = torch.cat((expo_candidate, expo_list.unsqueeze(0)), dim=0)
                expo_cost_candidate = torch.cat((expo_cost_candidate, expo_cost_list.unsqueeze(0)), dim=0)


                click_index_candidate = torch.cat(
                    (click_index_candidate, torch.zeros(list_length, dtype=torch.bool).unsqueeze(-2)), dim=0)

                if i>0:  # ignore the session don't start the first click
                    tgt_len=i
                    return {
                        'x_candidate': x_candidate,
                        "y_candidate": y_candidate,
                        "expo_candidate": expo_candidate,
                        "expo_cost_candidate": expo_cost_candidate,
                        "post_click_candidate": post_click_candidate,
                        "click_index_candidate": click_index_candidate,
                        "pre_click_utility_candidate": pre_click_utility_candidate,
                        "clicked_full_utility_candidate": clicked_full_utility_candidate,
                        "purchase_mask": purchase_mask,
                        "category": random_category,
                        "category_preferences": category_preferences,
                        "tgt_len": tgt_len}

                else:
                    return None



N_cate = 1   # Number of categories
binary_feature = 2  # Number of meaningful features per category
continous_feature =6  # Number of binary features in each product
num_cost_feature=4
num_postclick_feature=4
num_product = 5000  # Total number of products
num_consumer=10
list_length=10
preference_shock_var=1

num_product_per_cate = num_product/N_cate  # number of product per category
num_feature=N_cate*(binary_feature+continous_feature)+1

# Generate the Product Pool and Search Cost Pool
p_generator = ProductGenerator(N_cate, binary_feature, continous_feature, num_product)
products = p_generator.generate_products()
c_generator = ProductCostGenerator(num_product,num_cost_feature)
product_cost=c_generator.generate_costs()
pc_generator = ProductPostClickFeatureGenerator(num_product,num_postclick_feature)
product_postclick=pc_generator.generate_postclick_features()


pref_cost_generator=PreferCostCoeffGenerator(num_consumer,num_feature+num_cost_feature,num_feature,num_cost_feature,num_postclick_feature)
pref_cost,post_click_coeff=pref_cost_generator.generate_coeff()
preferences=pref_cost[:,:num_feature]
cost_coeff=pref_cost[:,num_feature:(num_feature+num_cost_feature)]



# Load RU module
module_ru = PreTrainedModuleRU()
module_ru.load_state_dict(torch.load("model_ru_parameter.pt"))  # Load the saved weights
module_ru.eval()  # Set to evaluation mode


# Sampler from a specific category
sampler = CategorySampler(products,product_cost,product_postclick, N_cate, num_product_per_cate)

# The Data set should inherent Dataset class
num_sessions=num_consumer*500
ConClickGen=ConsumerClick(preferences,cost_coeff,post_click_coeff, N_cate, num_feature,num_cost_feature, num_postclick_feature,sampler, module_ru,list_length,preference_shock_var)
simulated_sessions=ConClickGen.simulate_consecutive_sessions(num_sessions)





# write the data to pkl file
with open('simulated_sessions.pkl', 'wb') as f:
    pickle.dump(simulated_sessions, f)
