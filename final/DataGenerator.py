from numpy.random import choice
import pandas as pd
import osmnx as ox
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

class DataGenerator:
    def __init__(self,G=None,data=None,data_path=None,gdf_nodes_edges=None,node_masks=None,clip_dates=True):
        
        if data:
            self.data,self.station_data=data
        else:
            print("Loading data...")
            self.load_data(path=data_path,clip_dates=clip_dates)

        self.lat_factor=np.cos(self.data.lat.mean()*np.pi/180)

        if G:
            self.G=G
        else:
            print("Loading map...")
            self.load_map()

        if gdf_nodes_edges:
            self.gdf_nodes,self.gdf_edges=gdf_nodes_edges
        else:
            print("Creating GeoPandas DataFrame...")
            self.gdf()

        if 'closest_node' not in self.station_data.columns:
            print("Linking stations to nodes...")
            self.link_stations_to_nodes()

        if node_masks:
            self.node_masks,self.reduced_node_masks=node_masks
        else:
            print("Generating node masks...")
            self.generate_node_masks()

        self.T=[]

        #Save the parsed data for repeating the initialization fast
        self.last_init_data={'data':(self.data,self.station_data),
                             'G':self.G,
                             'gdf_nodes_edges': (self.gdf_nodes,self.gdf_edges),
                             'node_masks':(self.node_masks,self.reduced_node_masks)}

        

    def track_generator(self,n_tracks,sampling_rate=1,
                        max_iter_factor=10,
                        start_day=False,end_day=False,
                        start_h=0,end_h=23,
                        debug=False,**kwargs):
        """
        Generates a list of routes with points selected with the node_selector
        and with time simulated by the time_generator.
        This generates random days and hours within the input, not a sequence!

        Returns a list of n trajectories T.
        Each trajectory T[i] is a list of list elements in the format:
        [[timestamp, edge_start_node, edge_end_node, edge_osmid], ...]

        If there is not a path between selected nodes (not connected graph) it retries new nodes.
        A budget of n*max_iter_factor retries is set to avoid infinite loop.
        """

        if debug:
            print('Generation starting...')

        #Select the datetimes to use
        available_days=self.data["day"].unique()
        if start_day:
            available_days=available_days[available_days>=start_day]
        if end_day:
            available_days=available_days[available_days<=end_day]
        sub_data=self.data[self.data["day"].isin(available_days)]
        
        counts_per_day=sub_data.groupby("day")["counts"].sum()
        stations_per_day=sub_data.groupby("day")["id"].count()
        c_per_available_day=counts_per_day/stations_per_day
        p_per_available_day=self.probability_from_counts(c_per_available_day)
        
        available_hours=[*range(start_h,end_h+1)]

        self.T=[]
        max_counts=n_tracks*max_iter_factor
        counts=0
        while len(self.T)<n_tracks:
            try:
                day=choice(available_days,p=p_per_available_day)
                day_data=self.data[self.data["day"].values==day]
                day_data_available=day_data[day_data["hour"].isin(available_hours)]
                c_per_available_hour=day_data_available.groupby("hour")["counts"].sum()
                p_per_available_hour=self.probability_from_counts(c_per_available_hour)
                hour=choice(c_per_available_hour.index,p=p_per_available_hour)
                
                date=pd.Timestamp(year=day.year,month=day.month,day=day.day,hour=hour)
                
                orig, dest = self.node_selector(date,**kwargs)
                track=nx.shortest_path(self.G, orig, dest, weight='length')

                # if len(track)<2:
                #     print("orig,dest:",orig,dest)
                #     print("track:",track)

                t=date.timestamp()
                extended_track=self.time_generator(track,t,sampling_rate=sampling_rate,**kwargs)
                self.T.append(extended_track)

            except nx.exception.NetworkXNoPath:
                if debug:
                    print(f"-> Repeating {len(self.T)}th trajectory... (disconnected)")
                else:
                    pass
            except IndexError:
                if debug:
                    print(f"-> Repeating {len(self.T)}th trajectory... (same node)")
                else:
                    pass
            except ValueError:
                if debug:
                    print(f"-> Repeating {len(self.T)}th trajectory... (probability float overflow)")
                else:
                    pass

            counts+=1
            if counts==max_counts:
                print(f"WARNING: Not all tracks could be generated in {max_counts} iteratons.",
                      f"\nReturning {len(self.T)} tracks.")
                break

        if debug:
            print("Generation done.")
        self.T
        return self.T
    
    def edges_to_nodes(self,track):
        return [edge[0] for edge in track]

    def nodes_to_edges(self,track):
        return [(track[i],track[i+1]) for i in range(len(track)-1)]
    
    def gdf(self):
        self.gdf_nodes, gdf_edges = ox.graph_to_gdfs(self.G)
        self.gdf_edges=gdf_edges.droplevel(-1)
    
    def node_selector(self,date,**kwargs):
        """
        Selects two nodes at random from the map G with probability drawn from data
        """
        nodes,p=self.get_p(date)
        orig,dest = choice(nodes,p=p,size=2,replace=False)
        return orig, dest

    def get_p(self,date):
        """
        Retrieves the nodes and relative frequency of counts for a given date,
        using the node distance mask to the counters.
        """
        date_subdata=self.data[self.data["date"].values==date]
        counts=date_subdata.groupby("id")['counts'].sum()

        counts=counts.reindex(self.reduced_node_masks.columns).fillna(0)
        counts=(self.reduced_node_masks @ counts)/self.reduced_node_masks.sum(axis=1)
        counts=counts[~counts.isna()]

        nodes=counts.index.to_list()
        p=self.probability_from_counts(counts)
        return nodes,p

    def time_generator(self,track,start_t,sampling_rate=1,v0=20):
        """
        Adds timestemps to a list of nodes every samping_rate seconds
        simulating a movement at constant speed v0 [km/h]
        """   
        edge_list=self.nodes_to_edges(track)
        extended_track=[]
        edge_index=0
        t=start_t
        edge=edge_list[0]
        l_excess=0
        done=False
        while not done:
            
            l_excess+=sampling_rate*(v0/3.6)

            edge, l_excess, depth, done = self.l_excess_backtrack(edge_index,edge_list,l_excess)
            edge_index+=depth
            
            osmid=self.G[edge[0]][edge[1]][0]['osmid']
            x=[t,edge[0],edge[1],osmid]
            extended_track.append(x)

            t+=sampling_rate

        return extended_track

    def load_data(self,clip_dates=False,path=None):
        if not path:
            path='data/test/comptage-velo-donnees-compteurs.csv'
        raw_data=pd.read_csv(path,sep=";")

        data=raw_data.loc[:,["id","sum_counts","date","coordinates"]]
        data=data.rename(columns={"sum_counts":"counts"})
        #But some counters multiple subcounters. Aggregate!!

        data=data.drop(data[data.coordinates.isnull()].index,axis=0)
        #TODO: there are counters with isnull().sum()!=0 Always the same? Interpolate or equal 0?

        data=data.astype({"id":int})

        #Only care relative time (people always go to work at 8am, dont care about UTC)
        split_date=data["date"].str.split("+",expand=True)
        data["date"]=split_date[0]
        data["UTC+"]=split_date[1]
        data["date"]=pd.to_datetime(data['date'])

        data["day"]=data['date'].dt.date
        data["hour"]=data["date"].dt.hour #Since data.date.dt.minute.sum()=0, dont care about else

        data[["lat","lon"]]=data["coordinates"].str.split(",",expand=True).astype(float)
        data["coordinates"]=data["coordinates"].str.split(",",expand=False)
        data["coordinates"].apply(lambda x: [float(x[0]),float(x[1])])

        data["weekend"]=data["date"].dt.weekday>4

        unique_index=[data[data["id"]==_id].index[0] for _id in data["id"].unique()]
        station_data=data.loc[unique_index,['id','coordinates','lat','lon']]
        station_data=station_data.set_index("id")

        if clip_dates:
            #Clip from the left where not all stations implemented
            max_stations_day=data.groupby("day")["id"].count().idxmax()
            data=data[data["day"].values>max_stations_day]
            #Clip from right where counts stop being constant
            max_stations_by_date=data.groupby("date")["id"].count().max()
            counts_by_date=data.groupby("date")["id"].count()
            errors=counts_by_date[counts_by_date.values<max_stations_by_date/2] #When it drops to half, it is bad
            first_error_day=errors.index[0].date()
            data=data[data["day"].values<first_error_day]

        self.data=data
        self.station_data=station_data

    def link_stations_to_nodes(self):
        for station,X,Y in self.station_data[['lon','lat']].itertuples():
            d2=(self.gdf_nodes['x']-X)**2+self.lat_factor*(self.gdf_nodes['y']-Y)**2
            self.station_data.loc[station,'closest_node']=d2.idxmin()
    
    def generate_node_masks(self,d0=1):
        self.node_masks=pd.DataFrame(index=self.gdf_nodes.index,columns=self.station_data.index).fillna(0)
        for station,X,Y in self.station_data[['lon','lat']].itertuples():
            d2=(self.gdf_nodes['x']-X)**2+self.lat_factor*(self.gdf_nodes['y']-Y)**2
            d2*=(np.pi/180**2)
            d=6371*np.sqrt(d2)

            #Gaussian filter (more related to closest)
            g=np.exp(-np.power(d/(d0/3), 2.)/2.)
            #Absoute filter (cut all weights beyond d0)
            f=d<d0

            self.node_masks[station]=g*f
        self.reduced_node_masks=self.node_masks[(self.node_masks.sum(axis=1)!=0)]

    def load_map(self):
        self.G = ox.graph_from_place('Paris, France', network_type='bike',simplify=False,retain_all=False)

    def l_excess_backtrack(self,index,edges,l_excess,depth=0):
        """
        Given a list of edges and a length recursively traverses the
        edges until the length is covered

        Returns the next edge after traversing l_excess, current l_excess (<l),
        the depth of the backtrack return and a control bool that states if the track is finished.
        If l_excess is bigger than the remaining track length, returns last edge in the track with
        control bool True.
        """
        
        edge=edges[index]
        edge_info=self.G[edge[0]][edge[1]][0]
        l=edge_info["length"]
        if l_excess>l:
            l_excess-=l
            if (index+1)<len(edges):
                return self.l_excess_backtrack(index+1,edges,l_excess,depth+1)
            else:
                return edge, l_excess, depth, True
        else:
            return edge, l_excess, depth, False

    def probability_from_counts(self,counts):
        """
        Creates a probability from counts adding up to 1 (avoiding rounding errors)
        """
        p=np.array(counts)/sum(counts)
        p=list(p)
        imax=np.argmax(p)
        p[imax] = 1 - sum(p[0:imax])-sum(p[imax+1:]) #subtraction round errors smaller than normalization round errors.
        return p

    def plot_heatmap(self,show_non_traversed=True,
                     show_stations=False,
                     show_start=False,show_end=False,
                     bkg_color="black",dpi=100,**kwargs):
        
        edge_counts=self.gdf_edges.copy()
        edge_counts["counts"]=0
        
        start_nodes=[];end_nodes=[]
        for track in self.T:
            
            edges=list(set([(n1,n2) for _,n1,n2,_ in track]))
            edge_counts.loc[edges,"counts"]+=1

            start_nodes.append(track[0][1])
            end_nodes.append(track[-1][2])

        fig = plt.figure(figsize=(8,6),dpi=dpi)
        ax = plt.axes()

        edges_traversed=edge_counts[edge_counts["counts"].values>0]
        edges_not_traversed=edge_counts[edge_counts["counts"].values==0]

        #Plots:

        if show_non_traversed:
            edges_not_traversed.plot(ax=ax, legend=False, color='#999999',linewidth=0.2,zorder=-1)

        edges_traversed.plot(ax=ax, column="counts",linewidth=2/3,zorder=0,**kwargs)

        if show_stations:
            self.station_data.plot.scatter(x="lon",y="lat",ax=ax,c="red",s=2,zorder=2)#,label="stations")

        if show_start:
            self.gdf_nodes.loc[start_nodes].plot.scatter(ax=ax,x='x',y='y',c="lime",s=2,zorder=1)

        if show_end:
            self.gdf_nodes.loc[end_nodes].plot.scatter(ax=ax,x='x',y='y',c="green",s=2,zorder=1)
        
        # fig.colorbar(cm.ScalarMappable(norm=None, cmap='cool'),ax=ax,orientation='horizontal')

        ax.set(facecolor = bkg_color)
        plt.xticks([])
        plt.xlabel('')
        plt.yticks([])
        plt.ylabel('')
        ax.set_title("Heatmap of trajectories")#, fontsize=20)

        return fig,ax
