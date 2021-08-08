# DG Optimizer construction:
## parameters:
* population size                   :Learning/Model
* chromosome length                 :Learning/Model
* on dg count                       :Model
* solver_object                     :Model
* budget                            :Model
* dg_information_source             :Model
* facility/node information source  :Model
* distance information source if any:Model
* PM:=probability of mutation       :Learning
* PC:=probability of crossover      :Learning
* Number of Generations/Epochs      :Learning
* stopping threshold                :Learning
* mode                              :Learning

# DG_Solver_object construction:
## parameters:
* chromosome length: comes from DG info source
* on dg count:       set by user
* budget: budget for DG allocation
* dg_information_source: csv containing descriptions of DGs
* facility/node information source: description of nodes
* distance information source if any : csv matrix like where entry 
  ij is the distance from node i to dg j
  
# DG parameters and variables:
## parameters:
* source_file: csv containing DG info
  - assumed file column structure:
    + id 
    + output	
    + pos-x 
    + pos-y 
    + rated_power	
    + investment_cost	
    + o&m_cost	
    + excess_penetration_cost_LOW	
    + excess_penetration_cost_MEDIUM	
    + excess_penetration_cost_HIGH
* excessFactor: one of LOW, MEDIUM, HIGH
* initialize: boolean 
  - True: set all current outputs to 0
  - False: Use the outputs from the csv
* budget: value that a given assignment can not exceed
* assignment_options: number of facilities to power
* on_count: desired number of DGs to assign
* count_penalty: value indicating how much to scale how off the desired on count is

## DG variables/added columns
> The variables the DGs use to calculate costs
> Added columns:
> * costs: store costs for DG, aligned by index
> * current_output: DG current assigned output
> * assignments: lists of lists where the inner lists are a binary representation of 
>                a DGs assignment list. 1 indicates that node at index i is assigned to the DG, 0 no


# Node/Facility:
## parameters:
* sourcefile: csv file for facility info
  - expected columns:
    + id
    + demand
    + penalty
    + penalty
    * pos-x, pos-y
* distances: csv for distances from DG to node
* verbose: how much gets printed
* **kwargs:
  - ?
  

## variables:
* suppliedPower: how much power each node is supplied
* assignments: dictionary keyed on node index with value of index into DG Dataframe that is supplying that DG