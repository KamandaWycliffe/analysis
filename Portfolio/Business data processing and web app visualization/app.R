
#Install packages if missing
list.of.packages <- c("ggplot2", "grid", "gridExtra","dbplyr", "ltm", "data.table", "readxl", "ggpmisc", "ggpubr", "DT", "shinyWidgets", "knitr","plotly","tidyverse","shiny", "htmlTable", "jtools")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)


library(shiny)
library(tidyverse)
library(htmlTable)
library(knitr)
library(jtools)

# load 
collected_data <- read.csv('data/diamonds.csv')

# Pre-compute some variables to be used by app
not_numeric <- sapply(names(collected_data), function(x) !is.numeric(collected_data[[x]]))
data <- collected_data
#elect numeric variables
nums <- unlist(lapply(collected_data, is.numeric))
data_num <- collected_data[,nums]

# Define UI ----
ui <- fluidPage(
  
  titlePanel("Diamonds Dataset Explorer"),
  
  p("A simple Shiny APP suitable for exploratory data analysis, correlation, and regression analyses. The output in the
    'Summary' tab is based on the full dataset. The 
    controls in the sidebar are for the 'Plot','Data Overview', 'Correlaton', and 'Regression' tabs. The
    'Data Overview' tab shows a maximum of 15 rows of the dataset. The correlation shows the correlation test between the dependent and independnet atttibutes while the regerssion shows the 
    relationship between the price (target) and the selected independent attribute."),
  
  
  
  sidebarPanel(#checkboxGroupInput("variable", "Variables to show:",
                                  #c("Cylinders" = "cyl",
                                    #"Transmission" = "am",
                                   # "Gears" = "gear")),
    
    sliderInput("sampleSize", "Plot sample size (n)", min = 1, max = nrow(data),
                value = min(1000, nrow(data)), step = nrow(data) / 50, round = 0),
    radioButtons("smpType", "Plot sample type",
                 choices = list("Random k observations" = "random", "First k observations" = "first")),
    numericInput("smpseed", "Random Sample Seed", value = 10),
    
    selectInput("x", "X", names(data)),
    selectInput("y", "Y", c("None", names(data)), names(data)[[2]]),
    selectInput("x1", "independent", names(data_num)),
    selectInput("y1", "Dependent", c(names(data_num)), names(data_num)[[2]]),
    
    # only allow non-numeric variables for color
    selectInput("color", "Color (Filter by level)", c("None", names(data)[not_numeric])),
    
    p("Jitter and smoothing are used for numeric variables only
      are selected."),
    checkboxInput("jitter", "Jitter"),
    checkboxInput("smooth", "Smooth")
  ),
  
  mainPanel(
    
    # Output: Tabset
    tabsetPanel(type = "tabs",
                tabPanel("Plot", plotOutput("plot")),
                tabPanel("Data Overview", verbatimTextOutput("Overview")),
                tabPanel("Summary", verbatimTextOutput("summary")),
                tabPanel("Correlation Output", verbatimTextOutput("corr")),
                tabPanel("Regression Output", verbatimTextOutput("reg"))
    )
  )
)

# Define server logic ----
server <- function(input, output) {
  # Generate data summaries
  output$summary <- renderPrint({
    kable(summary(collected_data))
  })
  output$corr <- renderPrint({
    cor.test(data_num[[input$x1]], data_num[[input$y1]], 
             method = "pearson")
  })
  
  output$reg <- renderPrint({
    regr <- lm(data_num$price~data_num[[input$x1]])
    summary(regr)
    
    })
  #Plot regression 
  

  # get new dataset sample for plotting
  idx <- reactive({
    if (input$smpType == "first") {
      1:input$sampleSize
    } else {
      set.seed(input$smpseed)
      sample(nrow(collected_data), input$sampleSize)
    }
  })
  data <- reactive(collected_data[idx(), , drop = FALSE])
  
  # Get head of selected data
  output$Overview <- renderPrint({
    head(data(), n = 15)
  })
  
  
  # get plot type
  # * 2: both numeric variables
  # * 1: one numeric, one non-numeric variable
  # * 0: both non-numeric variables
  # * -1: only one variable provided
  plot_type <- reactive({
    if (input$y != "None")
      is.numeric(collected_data[[input$x]]) + is.numeric(collected_data[[input$y]])
    else
      -1
  })
  
  

  
  
  # Create plot
  output$plot <- renderPlot({
    if (plot_type() == 2) {
      # both numeric variables: scatterplot
      # also allow for color, jitter & smoothing
      pltt <- ggplot(data(), aes_string(x = input$x, y = input$y), color = "steelblue")
      pltt <- pltt + geom_smooth(method='lm')
      
      if (input$jitter)
        pltt <- pltt + geom_jitter(alpha = 0.5)
      else
        pltt <- pltt + geom_point(alpha = 0.5)
      
      if (input$smooth)
        pltt <- pltt + geom_smooth()
      
      # color change
      if (input$color != "None")
        pltt <- pltt + aes_string(color = input$color)
    } else if (plot_type() == 1) {
      # one numeric var, one character var: boxplot
      # allow color, don't allow jitter or smoothing
      pltt <- pltt <- ggplot(data(), aes_string(x = input$x, y = input$y)) + 
        geom_boxplot()
      
      # fill change
      if (input$color != "None")
        pltt <- pltt + aes_string(fill = input$color)
    } else if (plot_type() == 0) {
      # two character variables: heatmap
      # don't allow color, jitter or smoothing
      datafx <- reactive(data()[, c(input$x, input$y), drop = FALSE] %>%
                            group_by(across()) %>%
                            summarize(count = n())
      )
      pltt <- ggplot(datafx(), 
                  mapping = aes_string(x = input$x, y = input$y, fill = "count")) +
        geom_tile() +
        scale_fill_gradient(low = "#e7e7fd", high = "#1111dd")
    } else {
      # only one variable: univariate plot
      # allow color, don't allow jitter or smoothing
      pltt <- ggplot(data(), aes_string(x = input$x))
      
      if (is.numeric(collected_data[[input$x]]))
        pltt <- pltt + geom_histogram()
      else
        pltt <- pltt + geom_bar()
      
      # fill change
      if (input$color != "None")
        pltt <- pltt + aes_string(fill = input$color)
    }
    
    # add title
    if (plot_type() >= 0) {
      pltt <- pltt + labs(title = paste(input$y, "vs.", input$x))
    } else {
      pltt <- pltt + labs(title = paste("Distribution of", input$x))
    }
    
    # add styling
    pltt <- pltt + 
      theme_bw() +
      theme(plot.title = element_text(size = rel(1.8), face = "bold", hjust = 0.5),
            axis.title = element_text(size = rel(1.5)))
    
    print(pltt)
    
  }, height=600)
}

# Run the app ----
shinyApp(ui = ui, server = server)