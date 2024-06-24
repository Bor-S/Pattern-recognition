# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat
import random as rnd
from sklearn.neighbors import KNeighborsClassifier


if __name__ == "__main__":
    X_train = np.load("učni_vzorci.npy")
    y_train = np.load("učne_oznake.npy")
    X_test = np.load("testni_vzorci.npy")
    y_test = np.load("testne_oznake.npy")


# %%
### Razdelitev lokacij slik v listu v posamezne skupine za lažjo obravnavo naprej ###
razredNo = [[],[],[],[],[],[],[],[],[],[]] 
razredNoTest = [[],[],[],[],[],[],[],[],[],[]]

for i in range(len(y_train)):
    razredNo[y_train[i]].append(i)

for i in range(len(y_test)): # 0-9
    razredNoTest[y_test[i]].append(i) 


# %%
### Računanje značilk posamezne slike  ###
def calculate_image_features(class_number, sample_id, split):

    histogram = []   
    num_pixels_non_black = 0  
    razredNoTest = [[],[],[],[],[],[],[],[],[],[]]

    for i in range(len(y_test)):
        razredNoTest[y_test[i]].append(i)

    for y in range(28):
        for x in range(28):

            if split == 0:
                pixel_value = X_train[razredNo[class_number][sample_id], y, x]
            else:
                pixel_value = X_test[razredNoTest[class_number][sample_id], y, x]

            if pixel_value:  # Če pixel ni črn
                histogram.append(pixel_value)
                num_pixels_non_black += 1
    
    # Izračunaj značilke histograma
    mean_brightness = np.mean(histogram)
    variance = np.var(histogram)
    skewness = stat.skew(histogram)
    kurtosis = stat.kurtosis(histogram)

    return num_pixels_non_black, mean_brightness, variance, skewness, kurtosis




# %%
### Izračun značilk za vse vzorce ###
num_variables = 5

variables_train = np.zeros((10 * 6000, num_variables))
variables_test = np.zeros((10 * 1000, num_variables))

for i in range(10):  
    for j in range(6000):
        sample_index = razredNo[i][j]
        variables_train[razredNo[i][j]] = calculate_image_features(i, j, 0)
        
for i in range(10): 
    for j in range(1000):
        sample_index = razredNoTest[i][j]
        variables_test[sample_index] = calculate_image_features(i, j, 1)

# %%
### Brisanje značilk za testiranje različnih kombinacij ###
deleted_features = []  # Indeksi značilk [0,1,2,3,4]
X_train_features = np.delete(variables_train, (deleted_features), axis=1)
X_test_features = np.delete(variables_test, (deleted_features), axis=1)

# %%
### K-najbližjih sosedov klasifikator - 100 vzorcev na razred ###
num_features = 5 - len(deleted_features) 
num_samples_per_class = 100
random_sample_indices = []

X_train_small = np.zeros((10 * num_samples_per_class, 28, 28))
y_train_small = np.zeros((10 * num_samples_per_class,))
X_train_features = np.zeros((X_train_small.shape[0], num_features))
X_test_features = np.zeros((X_test.shape[0], num_features))

# 100 naključnih vzorcev
random_sample_indices = rnd.sample(range(0, 5999), 100)

for i in range(10):  # Za vsak razred
    for j in range(num_samples_per_class):
        sample_index = razredNo[i][random_sample_indices[j]]
        y_train_small[i * num_samples_per_class + j] = y_train[sample_index]
        X_train_features[i * num_samples_per_class + j] = variables_train[sample_index]

X_test_features = variables_test

K = 5  # Število sosedov

classifier = KNeighborsClassifier(n_neighbors=K)
classifier.fit(X_train_features, y_train_small)

# Testiranje klasifikatorja na testnih podatkih
y_test_predicted = classifier.predict(X_test_features)

# Preverjanje natančnosti klasifikatorja
correct_predictions = sum(y_test_predicted == y_test)
accuracy = correct_predictions / len(y_test)

print(f"Natančnost klasifikacije: {accuracy * 100:.2f}%")

# %%
def CalculateNeighborFeatures(class_num, sample_id, mode):
    directions = [[] for _ in range(8)]  # 8 smeri
    second_order_features = np.zeros(32)  # 32 značilk (4 na smer)

    for row in range(28):
        for col in range(28):
            # Trenutna vrednost piksla
            pixel_value = X_train[razredNo[class_num][sample_id], row, col] if mode == 0 else X_test[razredNoTest[class_num][sample_id], row, col]

            # Izrčunaj sosednje vrednosti, če trenutna vrednost ni črna
            if pixel_value != 0:
                for direction in range(8):
                    yOffset, xOffset = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)][direction]
                    newY, newX = row + yOffset, col + xOffset
                    if 0 <= newY < 28 and 0 <= newX < 28:
                        neighbor_value = X_train[razredNo[class_num][sample_id], newY, newX] if mode == 0 else X_test[razredNoTest[class_num][sample_id], newY, newX]
                        delta = int(neighbor_value) - int(pixel_value)
                        directions[direction].append(delta)

    # Izračunaj znčilke za vsako smer
    for i in range(8):
        second_order_features[i * 4] = np.mean(directions[i]) if directions[i] else 0  # Mean
        second_order_features[i * 4 + 1] = np.var(directions[i]) if directions[i] else 0  # Variance
        second_order_features[i * 4 + 2] = stat.skew(directions[i]) if directions[i] else 0  # Skewness
        second_order_features[i * 4 + 3] = stat.kurtosis(directions[i]) if directions[i] else 0  # Kurtosis

    return second_order_features


# %%
### Inicializacija polja za shranjevanje lastnosti drugega reda za učne podatke ###
features_train_second_order = np.zeros((10 * 6000, 32))
features_test_second_order = np.zeros((10 * 1000, 32))

for i in range(10):
    for j in range(6000):
        location = razredNo[i][j]
        feature_vector = CalculateNeighborFeatures(i, j, 0)
        features_train_second_order[location] = feature_vector 

for i in range(10):
    for j in range(1000):
        location = razredNoTest[i][j]
        feature_vector = CalculateNeighborFeatures(i, j, 1)
        features_test_second_order[location] = feature_vector 

# %%
### Združevanje značilk ###
total_features_count = 37 

# Inicializacija polj za združene značilke
combined_features_train = np.zeros((X_train.shape[0], total_features_count))
combined_features_test = np.zeros((X_test.shape[0], total_features_count))
combined_features_train = np.concatenate((variables_train, features_train_second_order), axis=1) 
combined_features_test = np.concatenate((variables_test, features_test_second_order), axis=1) 

# Shranjevanje 
np.save("combined_features_train.npy", combined_features_train)
np.save("combined_features_test.npy", combined_features_test)


# %%
### Brisanje značilk za testiranje različnih kombinacij ###

combined_features_train = np.load("combined_features_train.npy")
combined_features_test = np.load("combined_features_test.npy")

# Določi indekse značilk za brisanje
deleted_features = [] 

# Izbris določenih značilk iz naborov podatkov
combined_features_train_reduced = np.delete(combined_features_train, deleted_features, axis=1)
combined_features_test_reduced = np.delete(combined_features_test, deleted_features, axis=1)



# %%
### Klasifikator najbližjih sosedov z značilkami drugega reda - 100 vzorcev na razred ###

# Število značilk po brisanju
num_features = 37 - len(deleted_features)  
num_samples_per_class = 100
random_sample_indices = []

y_train_small = np.zeros((10 * num_samples_per_class,))
X_train_features_small = np.zeros((10 * num_samples_per_class, num_features))
X_test_features = np.zeros((X_test.shape[0], num_features))

# Izberi 100 naključnih vzorcev iz vsakega razreda
random_sample_indices = rnd.sample(range(0, 5999), 100)

for i in range(10):  # Za vsak razred
    for j in range(num_samples_per_class):
        lokacija = razredNo[i][random_sample_indices[j]]
        y_train_small[i * num_samples_per_class + j] = y_train[lokacija]
        X_train_features_small[i * num_samples_per_class + j] = combined_features_train_reduced[lokacija]

X_test_features = combined_features_test_reduced

K = 5  # Število sosedov

classifier = KNeighborsClassifier(n_neighbors=K)
classifier.fit(X_train_features_small, y_train_small)

# Testiranje klasifikatorja na testnih značilkah
y_test_predicted = classifier.predict(X_test_features)

# Preverjanje natančnosti klasifikatorja
pravilne_napovedi = np.sum(y_test_predicted == y_test)
natančnost = pravilne_napovedi / len(y_test)

print(f"Natančnost klasifikacije: {natančnost * 100:.2f}%")

# %%
### Standardizacija značilk ###
num_features = 37

combined_features_train_normalized = np.zeros((X_train.shape[0], num_features))
combined_features_test_normalized = np.zeros((X_test.shape[0], num_features))

mean_train = np.zeros((num_features))
mean_test = np.zeros((num_features))
std_dev_train = np.zeros((num_features))
std_dev_test = np.zeros((num_features))

## OSNOVNA NALOGA ##

# Izračun povprečja in standardnega odstopanja za vsako značilko
for i in range(num_features):
    mean_train[i] = np.mean(combined_features_train[:, i])
    mean_test[i] = np.mean(combined_features_test[:, i])
    std_dev_train[i] = np.std(combined_features_train[:, i])
    std_dev_test[i] = np.std(combined_features_test[:, i])
    # Normalizacija značilk tako, da bo povprečje 0 in standardno odstopanje 1
    combined_features_train_normalized[:, i] = (combined_features_train[:, i] - mean_train[i]) / std_dev_train[i]
    combined_features_test_normalized[:, i] = (combined_features_test[:, i] - mean_test[i]) / std_dev_test[i]

print(combined_features_train_normalized[0])
print(np.mean(combined_features_train_normalized, axis=0))
print(np.std(combined_features_train_normalized, axis=0))

"""
## DODATNA NALOGA ##

median_train = np.zeros((num_features))
median_test = np.zeros((num_features))
iqr_train = np.zeros((num_features))
iqr_test = np.zeros((num_features))

# Izračun mediane in medčetrtinskega razmaka (IQR) za vsako značilko
for i in range(num_features):
    median_train[i] = np.median(combined_features_train[:, i])
    median_test[i] = np.median(combined_features_test[:, i])
    iqr_train[i] = stat.iqr(combined_features_train[:, i])
    iqr_test[i] = stat.iqr(combined_features_test[:, i])
    # Normalizacija značilk tako, da bo mediana 0 in IQR 1
    combined_features_train_normalized[:, i] = (combined_features_train[:, i] - median_train[i]) / (iqr_train[i] if iqr_train[i] != 0 else 1)
    combined_features_test_normalized[:, i] = (combined_features_test[:, i] - median_test[i]) / (iqr_test[i] if iqr_test[i] != 0 else 1)

print(combined_features_train_normalized[0])
print(np.median(combined_features_train_normalized, axis=0))
print(stat.iqr(combined_features_train_normalized, axis=0))
"""


# %%
### Klasifikator za normalizirane značilke - 100 vzorcev na razred ###

num_features = 37
num_samples_per_class = 100
random_sample_indices = []

y_train_small = np.zeros((10 * num_samples_per_class,))
X_train_features_small = np.zeros((10 * num_samples_per_class, num_features))
X_test_features = np.zeros((X_test.shape[0], num_features))

random_sample_indices = rnd.sample(range(0, 5999), 100)
    
for i in range(10):
    for j in range(num_samples_per_class):
        location = razredNo[i][random_sample_indices[j]]
        y_train_small[i * num_samples_per_class + j] = y_train[location]
        X_train_features_small[i * num_samples_per_class + j] = combined_features_train_normalized[location]   

X_test_features = combined_features_test_normalized

K = 2  # Število sosedov

classifier = KNeighborsClassifier(n_neighbors=K)
classifier.fit(X_train_features_small, y_train_small)

# Testiranje klasifikatorja na normaliziranih testnih značilkah
y_test_predicted = classifier.predict(X_test_features)

# Preverjanje točnosti klasifikatorja
correct_predictions = np.sum(y_test_predicted == y_test)
accuracy = correct_predictions / len(y_test)

print(f"Točnost klasifikacije: {accuracy * 100:.2f}%")


# %%
def CalculateDistance(features, class_list, feature_index):
    # Izračun razdalje med razredi za določeno značilko
    num_classes = 10  # Število razredov
    samples_per_class_i = 100  # Vzorci na razred i
    samples_per_class_j = 100  # Vzorci na razred j
    total_distance = 0

    for i in range(num_classes):  # Za vsak razred
        for j in range(num_classes):  # Za vsak drugi razred
            if j != i:
                distance_sum = 0
                for k in range(samples_per_class_i):  # Za vsak vzorec v razredu i
                    for l in range(samples_per_class_j):  # Za vsak vzorec v razredu j
                        loc_feature_1 = class_list[i][k]  # Lokacija prve značilke
                        loc_feature_2 = class_list[j][l]  # Lokacija druge značilke

                        x1 = float(features[loc_feature_1, feature_index])
                        x2 = float(features[loc_feature_2, feature_index])

                        distance_sum += np.sqrt(pow(x1 - x2, 2))  # Seštevanje evklidskih razdalj
                        
                total_distance += distance_sum / (samples_per_class_i * samples_per_class_j)  # Povprečje razdalj za par razredov

    return total_distance  # Povprečje preko vseh parov razredov

# %%
# Izračun medrazredne razdalje
available_features = np.arange(37)  # Indeksi razpoložljivih značilk
dist_result_train = np.zeros(37)
dist_result_test = np.zeros(37)

# Izračunavanje razdalj
for feature_index in available_features:
    # Izračun razdalje za učne podatke
    dist_result_train[feature_index] = CalculateDistance(combined_features_train_normalized, razredNo, feature_index)
    
print(dist_result_train)  # Izpis razdalj za učne podatke

# %%
### Iskanje Najboljše Kombinacije - "NAPREJ" ###

final_features_train_forward = np.zeros(8).astype(int)  # Končne značilke
available_features = np.arange(37)
found_features_count = 0  # Število najdenih značilk

for i in range(len(final_features_train_forward)):

    feature_combinations = np.zeros((37 - found_features_count, found_features_count + 1))  # Vektor za shranjevanje kombinacij
    distance_combinations = np.zeros(37 - found_features_count)  # Vektor za shranjevanje izračunanih kombinacij

    # Izpolnite vektor kombinacij z najdenimi značilkami
    for j in range(37 - found_features_count):
        for x in range(found_features_count + 1):
            feature_combinations[j][x] = final_features_train_forward[x]

    # Dodajte drugačno značilko v vektor kombinacij
    for j in range(len(feature_combinations)):
        feature_combinations[j][found_features_count] = available_features[j]

    # Izračunajte in shranite razdalje za vsako kombinacijo
    for j in range(len(feature_combinations)):
        for x in range(len(feature_combinations[j])):
            distance_combinations[j] += dist_result_train[int(feature_combinations[j][x])]

    best_feature_index = np.argmax(distance_combinations)
    best_feature_value = dist_result_train[best_feature_index]

    # Dodajanje najboljše značilke, ki se ujema
    final_features_train_forward[found_features_count] = feature_combinations[best_feature_index][found_features_count]

    # Odstranjevanje razpoložljivih značilk za testiranje
    feature_to_remove = feature_combinations[best_feature_index][found_features_count]
    temp_available_features = available_features
    available_features = temp_available_features[temp_available_features != feature_to_remove]
    found_features_count += 1

print(final_features_train_forward)  # Izpis izbranih značilk


# %%
### Iskanje Najboljše Kombinacije - "NAZAJ" ###

final_features_train_backward = []  # Seznam za shranjevanje indeksov končno izbranih značilk
available_features = np.arange(37)  # Niz vseh indeksov značilk, ki so na voljo za izbor
remaining_features_count = 37  # Skupno število preostalih značilk

while remaining_features_count > 8:  # Nadaljujte, dokler ne ostane le 8 značilk

    feature_combinations = np.zeros((remaining_features_count, remaining_features_count - 1))  # Matrika za shranjevanje kombinacij
    distance_combinations = np.zeros(remaining_features_count)  # Niz za shranjevanje izračunanih kombinacij

    # Izpolnite matriko kombinacij s trenutnim naborom razpoložljivih značilk, pri čemer izključite eno značilko v vsaki iteraciji
    for j in range(remaining_features_count):
        count = 0
        for x in range(remaining_features_count):
            if j != x:
                feature_combinations[j][count] = available_features[x]
                count += 1

    # Izračunajte in shranite razdalje za vsako kombinacijo
    for j in range(remaining_features_count):
        for x in range(remaining_features_count - 1):
            distance_combinations[j] += dist_result_train[int(feature_combinations[j][x])]

    # Iskanje najmanj pomembne značilke (tiste, ki jo lahko odstranimo z najmanj vpliva)
    worst_feature_index = np.argmin(distance_combinations)
    worst_feature = available_features[worst_feature_index]

    # Odstranjevanje najmanj pomembne značilke iz razpoložljivih značilk
    available_features = np.delete(available_features, worst_feature_index)
    remaining_features_count -= 1

# Končni nabor značilk vključuje preostale značilke po postopku eliminacije nazaj
final_features_train_backward = available_features.tolist()

print(final_features_train_backward)  # Izpis izbranih značilk


# %%
### Klasifikator Najbližjih Sosedov z Najboljšimi Značilkami po Razredih - 100 Vzorcev na Razred ###

# Izbor značilk na podlagi izbora "naprej" ali "nazaj"
# Za izbor "naprej"
#selected_features_train = combined_features_train_normalized[:, final_features_train_forward]
#selected_features_test = combined_features_test_normalized[:, final_features_train_forward]
# Za izbor "nazaj"
selected_features_train = combined_features_train_normalized[:, final_features_train_backward]
selected_features_test = combined_features_test_normalized[:, final_features_train_backward]

num_features = 8
num_samples_per_class = 100
random_sample_indices = rnd.sample(range(0, 5999), 100)

# Inicializacija manjših nizov podatkov za KNN
X_train_small = np.zeros((10 * num_samples_per_class, num_features))
y_train_small = np.zeros(10 * num_samples_per_class)
X_test_small = np.zeros((X_test.shape[0], num_features))

# Izbor 100 naključnih vzorcev iz vsakega razreda
for i in range(10):
    for j in range(num_samples_per_class):
        location = razredNo[i][random_sample_indices[j]]
        y_train_small[i * num_samples_per_class + j] = y_train[location]
        X_train_small[i * num_samples_per_class + j] = selected_features_train[location]

# Uporaba izbranih značilk za testni nabor
X_test_small = selected_features_test

# Klasifikator Najbližjih Sosedov
K = 2
classifier = KNeighborsClassifier(n_neighbors=K)
classifier.fit(X_train_small, y_train_small)

# Preizkus klasifikatorja na značilkah testnih podatkov
y_test_predicted = classifier.predict(X_test_small)

# Preverjanje natančnosti klasifikatorja
correct_predictions = np.sum(y_test_predicted == y_test)
accuracy = correct_predictions / len(y_test)

print("Izbrane značilke (Naprej):", final_features_train_forward)
print("Izbrane značilke (Nazaj):", final_features_train_backward)
print(f"Natančnost klasifikatorja: {accuracy * 100:.2f}%")

# %%
### Dodatna naloga ###
# Izračunaj skupno razdaljo
def calculate_total_distance(features, class_list, feature_indices):
    total_distance = 0
    for feature_index in feature_indices:
        total_distance += CalculateDistance(features, class_list, feature_index)
    return total_distance

best_features_forward = []

# Implementacija zaporedne izbire značilk "naprej"
for _ in range(8):
    best_feature = None
    best_distance = 0
    for feature_index in range(37):
        if feature_index in best_features_forward:
            continue
        current_distance = calculate_total_distance(combined_features_train_normalized, razredNo, best_features_forward + [feature_index])
        if current_distance > best_distance:
            best_distance = current_distance
            best_feature = feature_index
    best_features_forward.append(best_feature)

print("Izbrane značilke (naprej):", best_features_forward)

# Implementacija zaporedne izbire značilk "nazaj"

best_features_backward = list(range(37))
while len(best_features_backward) > 8:
    worst_feature = None
    best_distance = 0
    for feature_index in best_features_backward:
        current_features = [f for f in best_features_backward if f != feature_index]
        current_distance = calculate_total_distance(combined_features_train_normalized, razredNo, current_features)
        if current_distance > best_distance:
            best_distance = current_distance
            worst_feature = feature_index
    if worst_feature is not None:
        best_features_backward.remove(worst_feature)

print("Izbrane značilke (nazaj):", best_features_backward)

# %%
### Testiranje klasifikatorja z izbranimi značilkami ###

selected_features = best_features_forward  # best_features_forward ali best_features_backward
X_train_selected = combined_features_train_normalized[:, selected_features]
X_test_selected = combined_features_test_normalized[:, selected_features]

classifier = KNeighborsClassifier(n_neighbors=2)
classifier.fit(X_train_selected, y_train)
y_test_predicted = classifier.predict(X_test_selected)

# Preverjanje natančnosti klasifikatorja
correct_predictions = np.sum(y_test_predicted == y_test)
accuracy = correct_predictions / len(y_test)
print(f"Natančnost klasifikacije z izbranimi značilkami: {accuracy * 100:.2f}%")



