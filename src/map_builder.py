import folium
from typing import List, Dict, Any

def generate_tour_map_html(
    start_point: List[float], 
    stops: List[Dict[str, Any]], 
    path_coords: List[List[float]],
    summary_text: str
) -> str:
   
    m = folium.Map(
        location=start_point,
        zoom_start=15,
        tiles="OpenStreetMap"
    )


    folium.Marker(
        location=start_point,
        popup=folium.Popup("<b>Start Location</b><br>Tour begins here", max_width=250),
        tooltip="Tour Starting Point",
        icon=folium.Icon(color="green", icon="play", prefix="fa")
    ).add_to(m)

  
    colors = ["blue", "purple", "orange", "darkred", "cadetblue"]
    for idx, stop in enumerate(stops):
        color = colors[idx % len(colors)]
        popup_html = f"""
        <div style="font-family: Arial, sans-serif; font-size: 13px; line-height: 1.4;">
            <h4 style="margin: 0 0 5px 0; color: #2c3e50;">#{stop['order']} {stop['name']}</h4>
            <b>Category:</b> {stop['category']}<br>
            <b>Stage Goal:</b> {stop['intent']}<br>
            <b>Allocated Stay:</b> {stop['dwell_time_min']} mins<br>
            <hr style="margin: 6px 0; border: 0; border-top: 1px solid #ccc;">
            <p style="margin: 0; font-size: 11px; color: #555;">{stop['description']}</p>
        </div>
        """
        folium.Marker(
            location=[stop["lat"], stop["lon"]],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"Stop #{stop['order']}: {stop['name']} ({stop['dwell_time_min']} min)",
            icon=folium.Icon(color=color, icon="info-sign")
        ).add_to(m)


    if path_coords and len(path_coords) > 1:
        folium.PolyLine(
            locations=path_coords,
            color="#2980b9",
            weight=5,
            opacity=0.85,
            tooltip="Walk Path"
        ).add_to(m)


    legend_html = f"""
     <div style="
         position: fixed; 
         top: 15px; right: 15px; width: 280px;
         background-color: white; z-index:9999; font-size:12px;
         border:2px solid #bdc3c7; border-radius: 8px; padding: 10px;
         box-shadow: 2px 2px 6px rgba(0,0,0,0.2);
         font-family: Arial, sans-serif;">
         <b style="font-size: 14px; color: #2c3e50;">UrbanTour Routing Engine</b><br>
         <hr style="margin: 5px 0;">
         <span>{summary_text}</span>
     </div>
     """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m.get_root().render()