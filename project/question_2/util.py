import sys

#http://www.scipy.org/
try:
	from numpy import dot, sqrt
	from numpy.linalg import norm
except:
	print("Error: Requires numpy from http://www.scipy.org/. Have you installed scipy?")
	sys.exit() 

def removeDuplicates(words_list:list):
	"""remove duplicates from a list"""
	return set((item for item in words_list))


def cosine(vector1, vector2):
	"""
	related documents j and q are in the concept space by comparing the vectors :
	cosine = ( V1 * V2 ) / ||V1|| x ||V2||

	param vector1: document_vector
	param vector2: query_vector
	"""
	# for the numpy array case
	# https://www.geeksforgeeks.org/how-to-calculate-cosine-similarity-in-python/
	if type(vector1) != list:
		cos_value = dot(vector1, vector2) / (norm(vector1, axis=1) * norm(vector2))
	# for the original case
	elif type(vector1) == list:
		# using slower way
		cos_value = float(dot(vector1, vector2) / (norm(vector1) * norm(vector2)))
	return cos_value

def euclidean(vector1, vector2) -> float:
	"""related documents j and q using euclidence distance"""
	if type(vector1) != list:
		#euclidean_value = sqrt(((vector2-vector1)**2).sum(axis = 1))
		euclidean_value = sqrt(((vector1-vector2)**2).sum(axis = 1))
	elif type(vector1) == list:
		euclidean_value = norm(vector1 - vector2)
	return euclidean_value

if __name__ == "__main__":
	pass