# neuropong
from NEAT import *

import random
import numpy as np
import pygame, pylab
from operator import itemgetter, attrgetter, methodcaller
import pickle


from pygame.locals import *

import networkx as nx

###########################################

class Paddle():

	def __init__(self):
		self.move = 0.0
		self.y = 0.0
		self.x = 0.0
		self.score = 0
	# not doing much with this at present, but this could become the basis for a better "classification" of the pongGame, and for a polymorphic classification of PongPhenotype as both Paddle and NEATnet.

class PongPhenotype(NEATnet):

	def __init__(self, name="", inOut=[5,1], mutation_rate=0.2, neurons=[], synapses=[], initial_pop=True, code=[]):
		
		
		self.gen = 0
		self.fitness = -0.0000001
		self.mutation_rate = mutation_rate

		self.move = 0.0
		self.y = 0.0
		self.x = 0.0
		self.score = 0
		
		self.wins = 0
		self.losses =0
		self.exp = 0
		self.offspring = 0
		self.mutations = 0
                self.inOut = inOut

		super(PongPhenotype, self).__init__(inOut, neurons, synapses, initial_pop) #, code=code)
		if name:
			self.name = name
	
	# def setLayers(self):
	# 	self.layers = [5,4,1]

	def sensorimotor(self, circle_x, circle_y, bar_y, speed_x, speed_y):


		inputs = [circle_x, circle_y, bar_y, speed_x, speed_y]

		outputs = self.feedForward(inputs)
		
		reaction = 2*np.tanh(outputs[0])

		return reaction

	def mate(self, other):

		EX  = 1.0
		DIS = 1.0
		W   = 1.0
		 
		# if self.speciesDistance(other, EX, DIS, W) >
		# 	return
		# we'll fill all this in later. Not sure what the values should look like yet. For the time being, every worm can mate with every other worm. No speciation yet. 
		print"\nWith EX = DIS = W = 1.0, the species distance between",self.name,"and",other.name,"=",self.speciesDistance(other, EX, DIS, W)
		print"They will now mate."
		child_schemata = {}
		child_schemata[self] = self.crossover(other)
		child_schemata[other] = other.crossover(self)
		
		# uncomment above line to have each mating pair make 2 children 
		children = []
		for parent in child_schemata.keys():
			
			
			child = PongPhenotype(inOut=parent.inOut, neurons=child_schemata[parent]["neurons"], synapses=child_schemata[parent]["synapses"], initial_pop=False)
			
			
			child.gen = max(self.gen,other.gen) +1
			#child.mutations = self.mutations + other.mutations
			child.mutation_rate = parent.mutation_rate
			
			if random.random() < child.mutation_rate:
				child.mutate(probWeightVsTopology=0.7)
				#child.mutations += 1

			child.name = random.choice(name_list)
			parent.offspring += 1
			
		

			children.append(child)
		
		print"\nmother: "+self.name
		for s in self.synapses: print s,
		print"\nfather: "+other.name
		for s in other.synapses: print s,
		print"\nchild: "+children[0].name
		for s in child_schemata[self]["synapses"]: print s,
		print"\nchild: "+children[1].name
		for s in child_schemata[self]["synapses"]: print s,
		print"\n"

		return children


class GeneticAlgorithm:

	def __init__(self, population_size=0, inOut=[5,1], mutation_rate=0.0, dueling=True, show_every=0, show_for=0):

		self.population = []
		self.population_size = population_size
		self.dueling = dueling
		self.show_every = show_every
		self.show_for = show_for
		self.iterations = 0
		self.makeGlobalNameList()
		self.iterations = 0

		self.populate(inOut, mutation_rate)

	def makeGlobalNameList(self):
		global name_list

		with open("data/names.txt") as f:
			name_list = f.read().splitlines()
		name_list = [name.rstrip() for name in name_list]
		with open("data/demonNames2.txt") as f:
			name_list2 = f.read().splitlines()
		name_list2 = [name.rstrip().upper() for name in name_list2]
		name_list += name_list2
		name_list = [name for name in name_list if 3 <= len(name) <= 10]
		name_list.sort()

	def populate(self, inOut, mutation_rate):

		for i in xrange(self.population_size - len(self.population)):

			name = random.choice(name_list)
			
			new_paddle = PongPhenotype(inOut=inOut, name=name, mutation_rate=mutation_rate)


			self.population.append(new_paddle)

	def statDisplay(self, player, top=15):
		pass
		# needs to be adapted. follow model in neuroworm

		BASICFONT = pygame.font.SysFont("droidmono",16)
		if player.fitness < 0:
			fit_string = "----"
		else:
			fit_string = (str(player.fitness)+" ")[:4]

		# genomeSurfA = BASICFONT.render('GENOME: %(genome)s' %{'genome':genomeNumber[0]}, True, (GENOMECOLOUR))
		# genomeSurfB = BASICFONT.render('        %(genome)s' %{'genome':genomeNumber[1]}, True, (GENOMECOLOUR))
		statSurf = BASICFONT.render('%(name)s FITNESS: %(fit)s   GENERATION: %(gen)s  NEURONS/SYNAPSES: %(ns)s  EXP: %(exp)s  WINS/LOSSES: %(wl)s' %{'name':(player.name+":"+" "*10)[:10], 'fit':fit_string, 'gen':player.gen, 'ns':str(len(player.neurons))+"/"+str(len(player.synapses)), 'exp':player.exp, 'wl':str(player.wins)+"/"+str(player.losses)}, True, (0, 200, 100))
		# # genomeRectA = genomeSurfA.get_rect()
		# # genomeRectB = genomeSurfB.get_rect()
		statRect = statSurf.get_rect()

		# # genomeRectA.top = (top)
		# # genomeRectB.top = (top+15)
		statRect.top	= (top)
		# # genomeRectA.x = (max(19., top/3.25))
		# # genomeRectB.x = (max(19., top/3.25))
		
		if top == 15:
			statRect.left    = 20.
		else:
			statRect.right = 620.

		# # genomeNum = genomeNumber[0]+genomeNumber[1]
		
		return [statSurf, statRect]



	def pongGame(self, paddle1, paddle2=None, inOut=[5,1]):
		# the default mode will have one of the two players be the prepackaged "ai", and the other our evolving perceptron. eventually, I'd like to have a new tournement mode of selection that involves the perceptrons playing against *each other*.

		if paddle2 is None:
			paddle2 = Paddle()

		if self.iterations % self.show_every <= self.show_for:
			slowdown = True
			#self.neuroGraph(paddle1) # doesn't display. not sure why.

		else:
			slowdown = False

		nuance = True

		if nuance:
			varSpeed 	= True
			varBounce 	= True # <== try next
			varServe 	= True
		else: 
			varSpeed = varBounce = varServe = False
		# Turning nuance on activates features like (1) variable serve angle, (2) bounce angle, relative to the point at which the paddle contacts the ball, and (3) variable paddle speed (whereby the perceptron has the option to move the paddle at regular speed, or 1.2*regular speed). It seems to take significantly longer for the perceptron to master a "nuanced" game of pong. 

		reaction = 0.0
		

		screen=pygame.display.set_mode((640,480),0,32)
		pygame.display.set_caption("neuropong!")

		WIDTH = 640.0
		HEIGHT = 480.0
		BALL = 15
		if slowdown:
			SPEED = 640
		else:
			SPEED = 16

		BALLCOLOUR   = [0, 255, 127]
		BAR1COLOUR   = [0,  250, 125]
		BAR2COLOUR   = [0,  100, 50]
		if self.dueling:
			BAR2COLOUR = BAR1COLOUR
		
		SCORECOLOUR  = [0,  75,  25]
		FRAMECOLOUR  = SCORECOLOUR
		GENOMECOLOUR = [0, 200, 100]


		BAR1COLOUR = tuple(BAR1COLOUR)
		BAR2COLOUR = tuple(BAR2COLOUR)
		BALLCOLOUR = tuple(BALLCOLOUR)
		SCORECOLOUR = tuple(SCORECOLOUR)
		GENOMECOLOUR = tuple(GENOMECOLOUR)
		FRAMECOLOUR = tuple(FRAMECOLOUR)
		SCORECOLOUR = tuple(SCORECOLOUR)

		HALFX, HALFY = WIDTH/2., HEIGHT/2.
		STARTSPEED = 250.
		MIDLINEX = HALFX+15 #???
		TOPY, BOTY = 5., 475.
		LEFTX, RIGHTX = 5., 675.
		BARLENGTH = 50.



		#Creating 2 bars, a ball and background.
		back = pygame.Surface((WIDTH,HEIGHT))
		#back = back.convert_alpha()
		### alpha attempt! 
		background = back.convert()
		#background.set_colorkey((0,0,0))
		background.fill((0,0,0))
		bar = pygame.Surface((10,50))
		bar1 = bar.convert()
		bar1.fill(BAR1COLOUR)
		bar2 = bar.convert()
		bar2.fill(BAR2COLOUR)
		circ_sur = pygame.Surface((BALL,BALL))
		circ = pygame.draw.circle(circ_sur,(BALLCOLOUR),(int(BALL/2),int(BALL/2)),int(BALL/2))
		circle = circ_sur.convert()
		circle.set_colorkey((0,0,0))

		# some definitions
		paddle1.x = 0+10. 
		paddle1.y = (HEIGHT/2.0)-BARLENGTH/2. 
		paddle1.move = 0.
		
		paddle2.x = WIDTH-20.
		paddle2.y = (HEIGHT/2.0)-BARLENGTH/2.
		paddle2.move = 0. 
		paddle2.score = 0
		# issue: i need to create a non-evo paddle2 object that can support all these attributes. 
		circle_x, circle_y = HALFX, HALFY
		
		speed_x, speed_y, speed_circ = STARTSPEED, STARTSPEED, STARTSPEED
		
		#clock and font objects
		clock = pygame.time.Clock()
		font = pygame.font.SysFont("monospace",40)
		
		############################################3
		
		statSurf, statRect = self.statDisplay(paddle1, 15)

		if self.dueling:
			statSurf2, statRect2 = self.statDisplay(paddle2, 455)

		if varServe: # let's try just turnign this one feature off, and leaving the rest "nuanced"
			serveAngles = range(-250,-50,1)+range(2,250,50)
		else:
			serveAngles = [-250, 250]
	 		
		paddle1.hits = 0
		paddle2.hits = 0

		mightScore = False
		waste = 0
		stalemateMeter = 0
		theyDidNotMove = [True, True]
		lastBar1_y, lastBar2_y = paddle1.y, paddle2.y
		hitCheck = 0
		loops = 0

		while True:
			loops += 1	
			# decides when to quit
			if (paddle1.score + paddle2.score + (waste/2) > 20) or (paddle1.hits + paddle2.hits > 200) or (stalemateMeter > 20):
				break			
					
			score1 = font.render(str(paddle1.score), True,(SCORECOLOUR))
			score2 = font.render(str(paddle2.score), True,(SCORECOLOUR))

			screen.blit(background,(0,0))
			frame = pygame.draw.rect(screen,(FRAMECOLOUR),Rect((5,5),(630,470)),2)
			middle_line = pygame.draw.aaline(screen,(FRAMECOLOUR),(MIDLINEX,TOPY),(MIDLINEX,BOTY))
			screen.blit(bar1,(paddle1.x,paddle1.y))
			screen.blit(bar2,(paddle2.x,paddle2.y))
			screen.blit(circle,(circle_x,circle_y))
			screen.blit(score1,(250.,210.))
			screen.blit(score2,(380.,210.))
			################################
			# print the stat and some stats at the top of the screen
			#################################
			screen.blit(statSurf, statRect)
			if self.dueling:
				screen.blit(statSurf2, statRect2)
			#######################################
			paddle1.y += paddle1.move
			if self.dueling:
				paddle2.y += paddle2.move
			######################################
			# movement of circle
			time_passed = clock.tick(1280)
			time_sec = time_passed / float(SPEED)
			# the movement of the ball
			circle_x += speed_x * time_sec
			circle_y += speed_y * time_sec
			#speed_circ = max(abs(speed_x), abs(speed_y))
			speed_circ = abs(speed_x)
			#speed_circ = np.sqrt(abs(speed_x)**2 + abs(speed_y)**2)
			train_speed = speed_circ * time_sec 
			#	trainer.
			if not self.dueling:
				if circle_x >= 305.: # if the ball is in the trainer's court...
					#if paddle2.y != circle_y + 7.5:
					if paddle2.y < circle_y + 7.5:
					    paddle2.y += train_speed
					if  paddle2.y > circle_y - 42.5:
					    paddle2.y -= train_speed
					# else:
					#     paddle2.y == circle_y + 7.5
				else:
					if paddle2.y < 215.:
						paddle2.y += train_speed
					elif paddle2.y > 215.:
						paddle2.y -= train_speed

			if paddle1.y >= HEIGHT-60.: paddle1.y = HEIGHT-60.
			elif paddle1.y <= 10. : paddle1.y = 10.
			if paddle2.y >= HEIGHT-60.: paddle2.y = HEIGHT-60.
			elif paddle2.y <= 10.: paddle2.y = 10.
				
			## BAR1 HITTING THE BALL ##
			if circle_x <= paddle1.x + 10.:
				if circle_y >= paddle1.y - 7.5 and circle_y <= paddle1.y + 42.5:
					#if slowdown:
					#	hit_sound.play()
					circle_x = 20.
					speed_x = -speed_x
					if varBounce:
						yDiff = 3*paddle1.move + random.randrange(10)#-((((circle_y-(paddle1.y+10))/BARLENGTH)*speed_y)+paddle1.move/2)
					else: 
						yDiff = 0
					speed_y += yDiff
					paddle1.hits += 1
					hitCheck += 1
					mightScore = True
					theyDidNotMove = [lastBar1_y-5 <= paddle1.y <=lastBar1_y+5, lastBar2_y-5 <= paddle2.y <=lastBar2_y+5]
					lastBar1_y = paddle1.y
					lastBar2_y = paddle2.y
			
					if all(theyDidNotMove):
						stalemateMeter += 1
					else:
						stalemateMeter = 0

			## BAR2 HITTING THE BALL ##
			if circle_x >= paddle2.x - 15.:
				if circle_y >= paddle2.y - 7.5 and circle_y <= paddle2.y + 42.5:
					#if slowdown:
					#	hit_sound.play()
					circle_x = WIDTH-75.
					speed_x = -speed_x
					if varBounce:
						yDiff = 3*paddle2.move + random.randrange(10)#-((((circle_y-(paddle2.y+10))/BARLENGTH)*speed_y)+paddle2.move/2)
					else:
						yDiff = 0
					speed_y += yDiff
					paddle2.hits += 1
					hitCheck += 1
					mightScore = True
					theyDidNotMove = [lastBar1_y-5 <= paddle1.y <=lastBar1_y+5, lastBar2_y-5 <= paddle2.y <=lastBar2_y+5]
					lastBar1_y = paddle1.y
					lastBar2_y = paddle2.y
					if all(theyDidNotMove):
						stalemateMeter += 1
					else:
						stalemateMeter = 0


			if circle_x < 5.:
				#if slowdown:	
                                    #miss_sound.play()
				if mightScore:
					paddle2.score += 1
					mightScore = False
				else:
					waste += 1
				circle_x, circle_y = HALFX, HALFY
				speed_y = random.choice(serveAngles)
				
				speed_x = -speed_x

				if paddle1.y > circle_y:
					# then paddle1.move should have been negative
					# so reaction should have been negative and abs(reaction) > 0.5

					errorSignal = [-(1.0 - abs(reaction))/300] # because the reaction was multiplied by a factor of 300 on the way out of the perceptron
				elif paddle1.y < circle_y:
					errorSignal = [+(1.0 - abs(reaction))/300]
					# then paddle1.move should have been positive
				## INSERT BACKPROP HERE!
				#paddle1.backProp(errorSignal)

				# LAMARCKIAN? 

				#paddle1.y,bar_2_y = 215., 215.
				# if the ball goes past the right edge of the playing field, then bar1 gets a point.
			elif circle_x > WIDTH-5.:
			#	if slowdown:
			#		miss_sound.play()
				if mightScore:
					paddle1.score += 1
					mightScore = False
				else:
					waste += 1
				circle_x, circle_y = HALFX, HALFY # put the ball back in the centre
				speed_y = random.choice(serveAngles)
				speed_x = -speed_x
				if paddle2.y < 215.:
					paddle2.y += train_speed
				elif paddle2.y > 215.:
					paddle2.y -= train_speed
			    #paddle1.y, paddle2.y = 215., 215.
			# if the ball hits the top or bottom, it bounces.
			if circle_y <= 10.:
			    speed_y = -speed_y
			    circle_y = 10.
			elif circle_y >= HEIGHT-22.5:
			    speed_y = -speed_y
			    circle_y = HEIGHT-22.5
			########################################
			#lastReaction = reaction

			# paddle11's move
			b1m = paddle1.sensorimotor((circle_x/WIDTH), (circle_y/HEIGHT), (paddle1.y/HEIGHT), (speed_x/250.0), speed_y/325.0)
			
			# if not varSpeed:
			# 	b1m = float(int(b1m)) # this neutralizes the possibility of a 1.5 or -1.5 result.
			paddle1.move = speed_circ*time_sec*b1m

			# paddle2's move
			if self.dueling:
				b2m = paddle2.sensorimotor(((WIDTH-circle_x)/WIDTH), (circle_y/HEIGHT), (paddle2.y/HEIGHT), ((-1*speed_x)/250.0), (speed_y/325.0))
				
				paddle2.move = speed_circ*time_sec*b2m
			
				# here's room to tinker too. what would happen if the AI had a slower paddle? faster?

			########################################

			if slowdown or True:
				pygame.display.update()
		###########################################
		# wins and losses
		if paddle1.score > paddle2.score:
			paddle1.wins += 1
			if self.dueling:
				paddle2.losses += 1
		elif paddle1.score < paddle2.score:
			paddle1.losses += 1
			if self.dueling:
				paddle2.wins += 1
		paddle1.exp += 1
		if self.dueling:
			paddle2.exp += 1
		###########################################
		# calculate fitness(es)
		##########################################
		totalScore = float(max(paddle1.score+paddle2.score, 1))
		totalHits = float(max(paddle1.hits+paddle2.hits, 1))
		scoreRatio = [paddle1.score/totalScore, paddle2.score/totalScore]
		hitRatio = [paddle1.hits/totalHits, paddle2.hits/totalHits]
		fitness = (3*scoreRatio[0] + 1*hitRatio[0])/4
			
		if not self.dueling:
			if paddle1.fitness > 0:
				fit1= (fitness+paddle1.fitness)/2
			else:
				fit1= fitness
		elif self.dueling:
			fitness2  = (2*scoreRatio[1] + 2*hitRatio[1])/4
			if fitness > fitness2:
				fitness = fitness   * (1+(fitness2))
			elif fitness2 > fitness:
				fitness2 = fitness2 * (1+(fitness)) 
			fitness, fitness2 = np.tanh(fitness), np.tanh(fitness2) # using tanh to normalize???
			if paddle1.fitness > 0:
				fit1= (fitness+paddle1.fitness)/2
			else:
				fit1= fitness
			if paddle2.fitness > 0:
				fit2 = (fitness2+paddle2.fitness)/2
			else:
				fit2 = fitness2
		###########################################
		if self.dueling:		
			e = "(L) "
		else:
				e = ""
		print e+(paddle1.name+" "*10)[:10],"Generation",paddle1.gen," Fitness =",(str(fit1)+"    ")[:6],"Wins/Losses:",str(paddle1.wins)+"/"+str(paddle1.losses)
		if self.dueling:
			print"(R) "+(paddle2.name+" "*10)[:10],"Generation",paddle2.gen," Fitness =",(str(fit2)+"    ")[:6],"Wins/Losses:",str(paddle2.wins)+"/"+str(paddle2.losses)
		if self.dueling: return [fit1, fit2]
		else: return fit1

	def tournement(self, mother=False):

		exclude = [mother]
		
		thePop = len(self.population)
		#print"TOURNEMENT"
		numberOfContenders = 2
		# grab a random group of potential mates

		contenders = []

		for i in range(numberOfContenders):
			contenders.append(random.choice([creature for creature in self.population if creature not in contenders+exclude]))

		for creature in contenders:
			creature.fitness = self.pongGame(creature)
		
		theWinner = [creature for creature in contenders if creature.fitness == max([creature.fitness for creature in contenders])][0]
		
		theDead = [creature for creature in contenders if creature != theWinner]
		
		assert len(theDead) == numberOfContenders - 1
	
		for creature in theDead:
			if len(self.population) > 12:
				self.population.remove(creature)

		#print theWinner.bits
		return theWinner

	def duel(self, mother=False):
		"""Selects two creatures from the populations, and has them directly compete. Returns the victor. Similar to tournement, except that the competition is direct, and so this function can only be used to call objective functions that allow for two genomes to participate simultaneously. Written with pong in mind."""
		contenders = []
		for i in range(2):
			contenders.append(random.choice([creature for creature in self.population if (creature not in contenders+[mother]) ]))
		contenders[0].fitness, contenders[1].fitness = self.pongGame(contenders[0],contenders[1])
		if contenders[0].fitness > contenders[1].fitness:
			self.population.remove(contenders[1])
			contenders.remove(contenders[1])
		elif contenders[0].fitness < contenders[1].fitness:
			self.population.remove(contenders[0])
			contenders.remove(contenders[0])
		else:
			d = random.choice((0,1))
			self.population.remove(contenders[d])
			contenders.remove(contenders[d])

		return contenders[0]

	def getElite(self,number):
		elite = []
		
		while len(elite) < number:
			fitnesses = [creature.fitness for creature in self.population if creature not in elite]
			e = [creature for creature in self.population if creature.fitness == max(fitnesses)]
			elite = elite + e

		if len(elite) > number:
			elite = elite[:number]
		return elite

	def neuroGraph(self, specimen ,node_size=2000, node_color='green', node_alpha=0.6, node_text_size=12, edge_color='black', edge_alpha=0.6, edge_tickness=1, edge_text_pos=0.3, text_font='droidmono'):
		
		pylab.clf()
		# create networkx (directed) graph
		G=nx.DiGraph()

		print"\nSpecimen:",specimen.name
		for synapse in specimen.synapses:
			print synapse,
		print "\n"
		# extract nodes from graph

		nodes = set(neuron.ID for neuron in specimen.neurons)
		#set([n1 for n1, n2 in graph] + [n2 for n1, n2 in graph])

		

		# add nodes
		for node in nodes:
			G.add_node(node)

		# words = ["L.EYE","NOSE", "TAIL", "R.EYE","PULSE","RUDDER","SPEED"]+range(7,len(nodes))
		# node_labels = {}
		# for i in xrange(len(nodes)):
		# 	node_labels[i] = str(words[i])
		nodelist = list(nodes)
		# nodelist.sort()
		# add edges
		for synapse in specimen.synapses:
			if synapse.enabled:
				G.add_edge(synapse.source.ID, synapse.target.ID, w=float(str(synapse.weight)[:10]))

		edge_labels = nx.get_edge_attributes(G,'w')
		#print edge_labels


		# draw graph
		pos = nx.shell_layout(G)
		nx.draw_networkx_edge_labels(G, pos, label_pos=0.5, font_weight="bold", bbox={"facecolor":"none", "edgecolor":"none"})
		#nx.draw_networkx_labels(G, pos, node_labels, font_size=10, font_color='k', alpha=edge_alpha)
		nx.draw_networkx_nodes(G, pos, nodelist[:5], alpha=node_alpha, node_size=node_size, node_color='g', node_shape='h')
		nx.draw_networkx_nodes(G, pos, nodelist[5:6],alpha=node_alpha, node_size=node_size, node_color='b', node_shape='h') 
		nx.draw_networkx_nodes(G, pos, nodelist[6:], alpha=node_alpha, node_size=node_size, node_color='c', node_shape='h')  # so^>v<dph8
		nx.draw_networkx_edges(G, pos, alpha=edge_alpha)

	
		nx.draw(G, pos, node_size=node_size)
		# show graph
		pylab.axis("off")
		pylab.title(specimen.name)
		pylab.draw()

	

	def runGeneticAlgorithm(self, iterations):
		
		for i in xrange(iterations):
			self.iterations += 1

			if not self.dueling:
				mother = self.tournement()
				father = self.tournement(mother = mother)
			elif self.dueling:
				mother = self.duel()
				father = self.duel(mother=mother)

			children = mother.mate(father)
			for child in children:
				self.population.append(child)

			fitTotal = [creature.fitness for creature in self.population if creature.fitness >= 0.0]
			if fitTotal != []:
				avgFit = sum(fitTotal)/float(len(fitTotal))
			else:
				avgFit = 0.0

			print"\nITERATION",self.iterations,"  AVG FITNESS:",avgFit
			neuron_count = [len(w.neurons) for w in self.population]
			avg_neuron_count = sum(neuron_count)/float(len(neuron_count))
			syn_count = [len(w.synapses) for w in self.population]
			avg_syn_count = sum(syn_count)/float(len(syn_count))
			max_syn_count = max(syn_count)
			max_neuron_count = max(neuron_count)
			print"Population:",len(self.population)
			print"Average number of neurons per worm:",avg_neuron_count
			print"Average number of synapses per worm:",avg_syn_count		
			print"Highest neuron count:",max_neuron_count
			print"Highest synapse count:", max_syn_count,"\n"


	def savePopulation(self):
		# HOW TO PICKLE
		if not getYesNoAnswer("\nSave population"):
			return
		filename=raw_input("Enter filename: ")
		popfile = open("populations/"+filename+".pop","w")
		pop = []
		self.population.sort(key=attrgetter('fitness'))
		for c in self.population:
			encoding = c.encode()
			pop.append(encoding)
		print pop
		pickle.dump(pop,popfile)
		popfile.close()

	def loadPopulation(self):
		while 1:
			filename = raw_input("Enter filename (without extension): ")
			try:
				popfile = open("populations/"+filename+".pop","r")
				pop = pickle.load(popfile)
				popfile.close()
				break
			except:
				print"Error loading population",filename+".pop"
				print"Check spelling and try again."

		for code in pop:
			creature = PongPhenotype(inOut=[5,1], code=code, initial_pop=False)
			self.population.append(creature)
		return pop 
			
			



#######################################################################


def getNumAnswer(nameOfValue, answerType=int):
	print nameOfValue+":"

	while 1:
		answer = raw_input(">> ")
		try:
			value = answerType(answer)
			break
		except ValueError:
			print"Unacceptable."
	return value


def getYesNoAnswer(nameOfValue):
	print nameOfValue+"? (Y/N)"
	yes = ["Y","YES","SURE","YEP"]
	no = ["N", "NO","NOPE","NO THANKS"]

	while 1:
		answer = raw_input(">> ")	
		if answer.upper() in yes:
			value = True
			break
		elif answer.upper() in no:
			value = False
			break
		else:
			print"Unacceptable."
			pass
	return value

#######################################################################

def main():
	pygame.init()
	# pygame.mixer.init()
	# global hit_sound, miss_sound
	#hit_sound = pygame.mixer.Sound("soundEffects/Bloob8Bit.wav")
	#miss_sound = pygame.mixer.Sound("")
	GA = GeneticAlgorithm()
	loadpop = getYesNoAnswer("Load population")

	if loadpop:
		
		GA.loadPopulation()
		GA.population_size = len(GA.population)
		skip = True
	else:
		skip = False
	
	default = getYesNoAnswer("Use default values")

	if default:
		GA.mutation_rate = 0.15
		if not skip:
			GA.population_size = 200
			GA.populate([5,1],GA.mutation_rate)
		GA.dueling = True
		GA.show_every = 1000
		GA.show_for = 3

	else:
		GA.mutation_rate = getNumAnswer("Mutation rate",float)
		if not skip:
			GA.population_size = getNumAnswer("Population size")
			GA.populate([5,1],GA.mutation_rate)
		GA.dueling = getYesNoAnswer("Have pongbots duel each other")
		GA.show_every = getNumAnswer("Interval at which to slow bouts down for viewing")
		GA.show_for = getNumAnswer("Number of bouts to show at a time")


	while 1:
		iterations = getNumAnswer("Enter number of iterations")
		
		GA.runGeneticAlgorithm(iterations)

		GA.savePopulation()
# this works with [5,2] inOut too. Nothing prevents us from having redundant neurons and synapses. This means that, in principle, we can deploy the same nets in different exercises. How well does a neuroworm play pong?

#################
main()


